import json
import re
import hashlib
import struct
import math
from itertools import tee
from operator import add
from typing import List, Tuple

import numpy as np
from scipy.integrate import quad as integrate
from bpemb import BPEmb
import fasttext

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark import RDD


class JsonDataUtil:
    """
    基于Spark的JSON数据治理工具类
    使用方式：
        util = JsonDataUtil()
        df = util.extract_json_fields(...)
        df = util.deduplicate_data(...)
    """

    SEED = 42
    RNG = np.random.RandomState(SEED)
    DTYPE = np.uint32
    MAX_HASH = 4_294_967_295
    MOD_PRIME = 4_294_967_291

    DEFAULT_LANG_MODEL = "lid.176.bin"
    DEFAULT_DEDUP_THRESHOLD = 0.9
    DEFAULT_NGRAM_SIZE = 5
    DEFAULT_MIN_LENGTH = 5
    DEFAULT_NUM_PERM = 250

    def __init__(self):
        self._ft_model = None
        self._multibpemb = None

    # ==================== 内部工具方法 ====================

    # -------------------- 语言检测 内部函数 --------------------
    def _load_lang_model(self, model_path):
        """加载并缓存语言模型"""
        if self._ft_model is None:
            self._ft_model = fasttext.load_model(model_path)
        return self._ft_model

    # -------------------- 去重 核心内部函数 --------------------
    def _get_bpemb(self):
        """缓存BPEmb分词模型"""
        if self._multibpemb is None:
            self._multibpemb = BPEmb(
                lang="multi", vs=1000000, dim=300,
                model_file='multi.wiki.bpe.vs1000000.model',
                segmentation_only=True
            )
        return self._multibpemb

    def _ngrams(self, sequence: List, n: int, min_length: int = 5):
        """生成N-Gram特征"""
        if len(sequence) < min_length:
            return []
        if len(sequence) < n:
            return [tuple(sequence)]
        iterables = tee(iter(sequence), n)
        for i, it in enumerate(iterables):
            for _ in range(i):
                next(it, None)
        return zip(*iterables)

    def _sha1_hash32(self, data):
        """32位SHA1哈希"""
        return struct.unpack("<I", hashlib.sha1(data).digest()[:4])[0]

    def _generate_edges(self, nodes: List[int]) -> List[Tuple[int, int]]:
        """生成连通图边"""
        if len(nodes) <= 1:
            return []
        min_node = min(nodes)
        return [(n, min_node) for n in nodes if n != min_node]

    def _small_star(self, edges: RDD):
        """Small-Star 算法"""
        def _map(e): return e if e[1] <= e[0] else (e[1], e[0])
        def _reduce(x):
            node, neighbors = x
            nodes = neighbors + [node]
            m = min(nodes)
            new = {(n, m) for n in nodes if n <= node and n != m}
            old = {(node, n) for n in neighbors}
            return list(new), len(new - old)

        neigh = edges.map(_map).groupByKey().mapValues(list)
        ec = neigh.map(_reduce).cache()
        change = 0 if ec.isEmpty() else ec.map(lambda x: x[1]).reduce(add)
        res = ec.flatMap(lambda x: x[0])
        ec.unpersist()
        return res, change

    def _large_star(self, edges: RDD):
        """Large-Star 算法"""
        def _map(e): return [(e[0], e[1]), (e[1], e[0])] if e[0] != e[1] else [e]
        def _reduce(x):
            node, neighbors = x
            m = min(neighbors + [node])
            new = {(n, m) for n in neighbors + [node] if n > node}
            old = {(node, n) for n in neighbors}
            return list(new), len(new - old)

        neigh = edges.flatMap(_map).groupByKey().mapValues(list)
        ec = neigh.map(_reduce).cache()
        change = 0 if ec.isEmpty() else ec.map(lambda x: x[1]).reduce(add)
        res = ec.flatMap(lambda x: x[0])
        ec.unpersist()
        return res, change

    def _alternating_algo(self, edges: RDD, max_iter=20):
        """交替迭代连通图算法"""
        lc = sc = float('inf')
        for _ in range(max_iter):
            edges, cl = self._large_star(edges)
            edges, cs = self._small_star(edges)
            if (cl == lc and cs == sc) or (cl == 0 and cs == 0):
                return edges, True
            lc, sc = cl, cs
        return edges, False

    def _generate_hash_values(self, idx, tokens, num_perm, ngram_size, min_len, hranges, perms):
        """生成MinHash特征分桶"""
        bpe = self._get_bpemb()
        grams = self._ngrams(list(bpe.encode(tokens)), ngram_size, min_len)
        tokens = {" ".join(g).encode() for g in grams}
        a, b = perms
        hv = np.array([self._sha1_hash32(t) for t in tokens], dtype=self.DTYPE)
        phv = np.bitwise_and(((hv * a[:, None] + b) % self.MOD_PRIME), self.MAX_HASH)
        min_hv = np.vstack([phv, np.full(num_perm, self.MAX_HASH, self.DTYPE)]).min(axis=0)
        return [(i, bytes(min_hv[s:e].byteswap().data), idx) for i, (s, e) in enumerate(hranges)]

    def _optimal_param(self, threshold, num_perm):
        """自动计算LSH最优分桶参数"""
        best_err, best = float('inf'), (1, 1)
        for b in range(1, num_perm + 1):
            max_r = num_perm // b
            for r in range(1, max_r + 1):
                fp = integrate(lambda s: 1 - (1 - s**r)**b, 0, threshold)[0]
                fn = integrate(lambda s: 1 - (1 - (1 - s**r)**b), threshold, 1)[0]
                err = 0.5 * fp + 0.5 * fn
                if err < best_err:
                    best_err, best = err, (b, r)
        return best

    # ==================== 对外 ====================

    # -------------------- JSONL字段提取 --------------------
    def extract_json_fields(self, spark, input_paths, field_list):
        """
        读取JSONL并提取嵌套字段
        :param spark: SparkSession 对象
        :param input_paths: str / list，输入路径
        :param field_list: str，字段列表，如 "uuid,text,doc_rate.total_score"
        :return: DataFrame，保留原始 dict/list 结构，不序列化
        """
        # 1. 字段字符串转列表
        if isinstance(field_list, str):
            field_list = [f.strip() for f in field_list.split(",") if f.strip()]

        # 2. 解析路径
        if isinstance(input_paths, str):
            paths = [p.strip() for p in input_paths.split(",")]
        else:
            paths = input_paths

        # 3. 原生读取 JSON（稳定、递归、不崩溃）
        df_raw = spark.read \
            .option("multiline", "false") \
            .option("recursiveFileLookup", "true") \
            .option("allowBackslashEscaping", "true") \
            .option("mode", "PERMISSIVE") \
            .json(paths)

        # 4. 提取字段，保留原始结构（关键！）
        select_cols = []
        for field in field_list:
            alias_name = field.replace(".", "_")
            # 直接取字段，不做任何to_json，保持 dict/list 原样
            select_cols.append(F.col(field).alias(alias_name))

        df_final = df_raw.select(*select_cols)

        # 调试输出
        print(f"[INFO] 读取数据条数: {df_final.count()}")
        print(f"[INFO] 提取字段: {field_list}")
        df_final.printSchema()

        return df_final

    # -------------------- 语言检测打标 --------------------
    def add_language_column(self, df, text_col, output_col="language", model_path=None):
        """
        给DF添加语言检测列（zh/en/other）
        :param df: DataFrame，输入数据集
        :param text_col: str，文本列名
        :param output_col: str，输出语言列名
        :param model_path: str，fasttext 模型路径
        :return: DataFrame，新增语言列后的数据集
        """
        model_path = model_path or self.DEFAULT_LANG_MODEL

        def _detect(text):
            if not text or len(text.strip()) < 3:
                return "other"
            m = self._load_lang_model(model_path)
            clean = text.replace("\n", " ")[:2000]
            lang = m.predict(clean, k=1)[0][0].replace("__label__", "")
            return "zh" if lang == "zh" else "en" if lang == "en" else "other"

        return df.withColumn(output_col, F.udf(_detect, StringType())(F.col(text_col)))

    # -------------------- 文本去重 --------------------
    def deduplicate_data(self, df, dedup_col, threshold=None, num_perm=None, ngram_size=None, min_length=None):
        """
        文本去重（MinHashLSH + 连通图）
        :param df: DataFrame，输入数据集
        :param dedup_col: str，去重依据的文本列名
        :param threshold: float，相似度阈值
        :return: DataFrame，去重后数据集
        """
        threshold = threshold or self.DEFAULT_DEDUP_THRESHOLD
        num_perm = num_perm or self.DEFAULT_NUM_PERM
        ngram_size = ngram_size or self.DEFAULT_NGRAM_SIZE
        min_length = min_length or self.DEFAULT_MIN_LENGTH

        B, R = self._optimal_param(threshold, num_perm)
        hranges = [(i * R, (i + 1) * R) for i in range(B)]
        perms = (
            self.RNG.randint(1, self.MOD_PRIME, size=num_perm, dtype=self.DTYPE),
            self.RNG.randint(0, self.MOD_PRIME, size=num_perm, dtype=self.DTYPE)
        )

        df = df.withColumn("__id__", F.monotonically_increasing_id())
        rdd = df.select("__id__", dedup_col).rdd

        buckets = rdd.flatMap(lambda x: self._generate_hash_values(
            x["__id__"], x[dedup_col][:10000], num_perm, ngram_size, min_length, hranges, perms
        )).groupBy(lambda x: (x[0], x[1])).mapValues(lambda xs: [x[2] for x in xs]).cache()

        edges = buckets.flatMap(self._generate_edges).distinct().cache()
        if edges.isEmpty():
            return df.drop("__id__")

        edges, _ = self._alternating_algo(edges)
        self_edges = edges.values().distinct().map(lambda x: (x, x))
        all_edges = edges.union(self_edges)

        return df.join(
            spark.createDataFrame(all_edges, ["__id__", "__c"]),
            on="__id__", how="left"
        ).filter(
            F.col("__c").isNull() | (F.col("__c") == F.col("__id__"))
        ).drop("__id__", "__c")

    # -------------------- 文件夹去污 --------------------
    def decontaminate_data(self, df, clean_col, pollute_dir, spark):
        """
        函数B：按污染文件夹去污
        :param df: DataFrame，输入数据集
        :param clean_col: str，需要去污的列名
        :param pollute_dir: str，污染词文件夹路径
        :param spark: SparkSession 对象
        :return: DataFrame，去污后数据集
        """
        if not pollute_dir:
            return df

        query_df = spark.read.option("multiline", "false") \
            .option("recursiveFileLookup", "true") \
            .json(pollute_dir)

        # 清洗query（去重、非空、长度>20）
        query_df = query_df.dropna(subset=["query"]) \
                        .drop_duplicates(["query"]) \
                        .filter(F.length(F.col("query")) > 20)

        query_count = query_df.count()
        print(f"[INFO] 有效去污query数量：{query_count}")

        if query_count == 0:
            print("[WARN] 无有效query，跳过去污")
            return df

        # 广播query
        query_list = query_df.select("query").rdd.map(lambda x: x[0]).collect()
        broadcast_queries = spark.sparkContext.broadcast(query_list)

        # UDF 匹配
        def is_contains_query(text):
            if not text:
                return False
            return any(q in text for q in broadcast_queries.value)

        contains_udf = F.udf(is_contains_query, BooleanType())

        total_docs = df.count()
        cleaned_df = df.filter(~contains_udf(F.col(clean_col)))
        cleaned_count = cleaned_df.count()

        print("=" * 60)
        print(f"去污前：{total_docs}")
        print(f"去污后：{cleaned_count}")
        print(f"过滤：{total_docs - cleaned_count}")
        print("=" * 60)

        return cleaned_df

    # -------------------- 字符串去污 --------------------
    def filter_by_contamination_str(self, df, clean_col, pollute_str):
        """
        函数C：按指定字符串过滤污染数据
        :param df: DataFrame，输入数据集
        :param clean_col: str，需要过滤的列名
        :param pollute_str: str，污染关键词/字符串
        :return: DataFrame，过滤后数据集
        """
        if not pollute_str:
            return df

        @F.udf(BooleanType())
        def _has(s):
            return s is not None and pollute_str in s

        return df.filter(~_has(F.col(clean_col)))

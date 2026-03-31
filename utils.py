import json
from pyspark.sql import functions as F
from pyspark.sql.types import *

def extract_json_fields(spark, input_paths, field_list):
    """
    读取 JSONL 文件并提取指定字段（支持嵌套，支持复杂结构）
    
    参数:
        spark: SparkSession 对象
        input_paths: 输入路径列表（list），或者逗号分隔的字符串
        field_list: 要提取的字段列表，支持嵌套，例如:
                    ["uuid", "text", "doc_rate", "doc_rate.total_score"]
    
    返回:
        提取后的 DataFrame
    """
    
    def _read_text():
        if isinstance(input_paths, str):
            paths = [p.strip() for p in input_paths.split(",")]
        else:
            paths = input_paths
            
        df = spark.read \
            .option("recursiveFileLookup", "true") \
            .option("pathGlobFilter", "*.jsonl") \
            .text(paths)
        return df

    def _get_nested_value(obj, path_parts):
        current = obj
        for part in path_parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    def _safe_serialize(value):
        if value is None:
            return None
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    def _process_line(line):
        try:
            obj = json.loads(line)
            result = []
            for field in field_list:
                path_parts = field.split(".")
                value = _get_nested_value(obj, path_parts)
                result.append(_safe_serialize(value))
            return tuple(result)
        except:
            return tuple([None] * len(field_list))

    df_raw = _read_text()
    

    schema = StructType()
    for field in field_list:
        # 把点号换成下划线作为列名，例如 doc_rate.total_score -> doc_rate_total_score
        col_name = field.replace(".", "_")
        schema.add(col_name, StringType())
    

    process_udf = F.udf(_process_line, schema)
    df_result = df_raw.select(process_udf(F.col("value")).alias("data")) \
        .select("data.*")
    
    return df_result

def add_language_column(df, text_col, output_col="language", model_path="lid.176.bin"):
    """
    给 DataFrame 添加语言检测列
    
    参数:
        df: 输入 DataFrame
        text_col: 用于检测语言的列
        output_col: 新增列，默认为 "language"
        model_path: fasttext 模型路径，默认为 "lid.176.bin"
    
    返回:
        新增语言列后的 DataFrame
    """
    
    def _load_model():
        global _ft_model
        if _ft_model is None:
            _ft_model = fasttext.load_model(model_path)
        return _ft_model
    
    def _detect_lang(text):
        if not text or len(text.strip()) < 3:
            return "other"
        
        model = _load_model()
        clean = text.replace("\n", " ")[:2000]
        pred = model.predict(clean, k=1)
        lang = pred[0][0].replace("__label__", "")
        
        if lang == "zh":
            return "zh"
        elif lang == "en":
            return "en"
        else:
            return "other"
    
    lang_udf = F.udf(_detect_lang, StringType())
    return df.withColumn(output_col, lang_udf(F.col(text_col)))

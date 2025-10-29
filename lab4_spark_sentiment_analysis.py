# test/lab5_spark_sentiment_analysis.py
import pytest
try:
    import pyspark
    SPARK_AVAILABLE = True
except Exception:
    SPARK_AVAILABLE = False

@pytest.mark.skipif(not SPARK_AVAILABLE, reason="PySpark not installed in this environment")
def test_spark_pipeline_basic():
    # placeholder spark test: this will only run if pyspark is available
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.master("local[2]").appName("test").getOrCreate()
    df = spark.createDataFrame([("good",1),("bad",0)], ["text","label"])
    assert df.count() == 2

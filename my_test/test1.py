import pyspark.sql.functions as F
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.feature import Bucketizer
from pyspark.sql import DataFrame
from typing import Iterable
import pandas as pd
from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local[*]").appName("FiratApp").getOrCreate()
# CUSTOM TRANSFORMER ----------------------------------------------------------------


# SAMPLE DATA -----------------------------------------------------------------------
df = pd.DataFrame({'ball_column': [0,1,2,3,4,5,6],
                   'keep_the': [6,5,4,3,2,1,0],
                   'hall_column': [2,2,2,2,2,2,2] })
df = spark.createDataFrame(df)

df.show()
#df.foreach(print)
from pyspark import keyword_only
from pyspark.ml import Transformer,Estimator
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.sql.functions import udf


class ReverseTransformer(Transformer, HasInputCol, HasOutputCol):
 
   @keyword_only
   def __init__(self, inputCol=None, outputCol=None):
       super(ReverseTransformer, self).__init__()
       kwargs = self._input_kwargs
       self.setParams(**kwargs)
 
   @keyword_only
   def setParams(self, inputCol=None, outputCol=None):
       kwargs = self._input_kwargs
       return self._set(**kwargs)
 
   def _transform(self, dataset):
       reverse = udf(lambda sentence: sentence[::-1])(dataset[self.getInputCol()])
       return dataset.withColumn(self.getOutputCol(), reverse)


from pyspark.sql import SparkSession
from pyspark.ml import Pipeline


session = SparkSession.builder.master("local[*]").getOrCreate()

df = session.createDataFrame([("foo bar",), ("hello world",)]).toDF("sentence")
reverse = ReverseTransformer(inputCol="sentence", outputCol="reversed")

pipeline = Pipeline(stages=[reverse])
model = pipeline.fit(df)

model.transform(df).show()

#model.save("notebook/model/spark_model")


将除ETL的部分都视为模型的组成部分，通过调整参数，分配各部分的贡献。

An important task in ML is model selection, or using data to find the best model or parameters for a given task. This is also called tuning. Tuning may be done for individual Estimators such as LogisticRegression, or for entire Pipelines which include multiple algorithms, featurization, and other steps. Users can tune an entire Pipeline at once, rather than tuning each element in the Pipeline separately.

MLlib supports model selection using tools such as CrossValidator and TrainValidationSplit. These tools require the following items:

Estimator: algorithm or Pipeline to tune
Set of ParamMaps: parameters to choose from, sometimes called a “parameter grid” to search over
Evaluator: metric to measure how well a fitted Model does on held-out test data
At a high level, these model selection tools work as follows:

They split the input data into separate training and test datasets.
For each (training, test) pair, they iterate through the set of ParamMaps:
For each ParamMap, they fit the Estimator using those parameters, get the fitted Model, and evaluate the Model’s performance using the Evaluator.
They select the Model produced by the best-performing set of parameters.
The Evaluator can be a RegressionEvaluator for regression problems, a BinaryClassificationEvaluator for binary data, or a MulticlassClassificationEvaluator for multiclass problems. The default metric used to choose the best ParamMap can be overridden by the setMetricName method in each of these evaluators.

To help construct the parameter grid, users can use the ParamGridBuilder utility. By default, sets of parameters from the parameter grid are evaluated in serial. Parameter evaluation can be done in parallel by setting parallelism with a value of 2 or more (a value of 1 will be serial) before running model selection with CrossValidator or TrainValidationSplit. The value of parallelism should be chosen carefully to maximize parallelism without exceeding cluster resources, and larger values may not always lead to improved performance. Generally speaking, a value up to 10 should be sufficient for most clusters.
import tensorflow as tf
import shutil
import math
from datetime import datetime
import multiprocessing
from tensorflow.python.feature_column import feature_column
import schema

print(tf.__version__)
sess = tf.Session()

MODEL_NAME = 'reg-model-1_2'
root_p = "E:/my_proj/fog_recognition/recognition_pre_fog_tf/recognition_pre_fog/"
TRAIN_DATA_FILES_PATTERN = root_p +'data/train-*.csv'
VALID_DATA_FILES_PATTERN = root_p + 'data/valid-*.csv'
TEST_DATA_FILES_PATTERN = root_p + 'data/test-*.csv'

RESUME_TRAINING = False
PROCESS_FEATURES = True
MULTI_THREADING = True

HEADER = ['key', 'x', 'y', 'alpha', 'beta', 'target']
HEADER_DEFAULTS = [[0], [0.0], [0.0], ['NA'], ['NA'], [0.0]]
NUMERIC_FEATURE_NAMES = ['x', 'y']
CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY = {'alpha':['ax01', 'ax02'], 'beta':['bx01', 'bx02']}
CATEGORICAL_FEATURE_NAMES = list(CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY.keys())
FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES
TARGET_NAME = 'target'
UNUSED_FEATURE_NAMES = list(set(HEADER) - set(FEATURE_NAMES) - {TARGET_NAME})
print("Header: {}".format(HEADER))
print("Numeric Features: {}".format(NUMERIC_FEATURE_NAMES))
print("Categorical Features: {}".format(CATEGORICAL_FEATURE_NAMES))
print("Target: {}".format(TARGET_NAME))
print("Unused Features: {}".format(UNUSED_FEATURE_NAMES))

from typing import ByteString, Dict, Tuple, Optional, Any


def parse_csv_row(csv_row: ByteString)->Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    """get a string tensor"""
    columns = tf.decode_csv(csv_row, record_defaults=HEADER_DEFAULTS)
    features: Dict[str, tf.Tensor] = dict(zip(HEADER, columns))
    for column in UNUSED_FEATURE_NAMES:
        features.pop(column)
    target = features.pop(TARGET_NAME)
    return features, target


def process_features(features: Dict[str, tf.Tensor])->Dict[str, tf.Tensor]:
    features["x_2"] = tf.square(features['x'])
    features["y_2"] = tf.square(features['y'])
    features["xy"] = tf.multiply(features['x'], features['y'])  # features['x'] * features['y']
    features['dist_xy'] = tf.sqrt(tf.squared_difference(features['x'], features['y']))
    return features


def csv_input_fn(file_name_pattern: str, mode: str=tf.estimator.ModeKeys.EVAL, batch_size: int=200,
                 num_epochs: Optional[int]=None, skip_header_lines: int=0)->Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False
    input_file_names: tf.Tensor = tf.matching_files(pattern=file_name_pattern)
    dataset = tf.data.TextLineDataset(input_file_names)
    dataset = dataset.skip(skip_header_lines)
    dataset = dataset.map(parse_csv_row)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=2 * batch_size + 1)
    if PROCESS_FEATURES:
        dataset = dataset.map(lambda features, target: (process_features(features), target))
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()
    feaures, target = iterator.get_next()
    return feaures, target


CONSTRUCTED_NUMERIC_FEATURES_NAMES = ['x_2', 'y_2', 'xy', 'dist_xy']


def get_feature_columns()->Dict[str, Any]:
    # 将各种列赋予类型，根据类型可以有响应处理
    all_numeric_feature_names = NUMERIC_FEATURE_NAMES + CONSTRUCTED_NUMERIC_FEATURES_NAMES
    numeric_columns: Dict[str, Any] = {
        feature_name: tf.feature_column.numeric_column(feature_name) for feature_name in all_numeric_feature_names}

    categorical_column_with_vocabulary: Dict[str, Any] = {
        item[0]: tf.feature_column.categorical_column_with_vocabulary_list(item[0], item[1])
        for item in CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY.items()}
    feature_columns = {}
    if numeric_columns is not None:
        feature_columns.update(numeric_columns)

    if categorical_column_with_vocabulary is not None:
        feature_columns.update(categorical_column_with_vocabulary)
    # 之所用用字典，方便后面按照名称索引进一步处理，因为列还没有进一步转化
    feature_columns['alpha_X_beta'] = tf.feature_column.crossed_column([feature_columns['alpha'], feature_columns['beta']], 4)

    return feature_columns


print("Feature Columns: {}".format(get_feature_columns()))


def create_estimator(run_config, hparams)->tf.estimator.Estimator:
    # 检查字段并对字段做按类别做出进一步处理
    FEATURE_COLUMNS = list(get_feature_columns().values())
    dense_columns = list(filter(lambda column: isinstance(column, feature_column._NumericColumn),FEATURE_COLUMNS))

    categorical_columns = list(
        filter(lambda column: isinstance(column, feature_column._VocabularyListCategoricalColumn) |
                              isinstance(column, feature_column._BucketizedColumn), FEATURE_COLUMNS))

    # convert categorical columns to indicators,独热码化
    indicator_columns = list(map(lambda column: tf.feature_column.indicator_column(column), categorical_columns))

    estimator = tf.estimator.DNNRegressor(
        feature_columns=dense_columns + indicator_columns,
        hidden_units=hparams.hidden_units,
        optimizer=tf.train.AdamOptimizer(),
        activation_fn=tf.nn.elu,
        dropout=hparams.dropout_prob,
        config=run_config
    )

    print("")
    print("Estimator Type: {}".format(type(estimator)))
    print("")
    return estimator

# ##########################准备试验数据#############################


#  #####定义试验参数#######
EVAL_AFTER_SEC = 15
NUM_EPOCHS = 1000
BATCH_SIZE = 500
TRAIN_SIZE = 12000
TOTAL_STEPS = (TRAIN_SIZE/BATCH_SIZE)*NUM_EPOCHS
hparams = tf.contrib.training.HParams(
    num_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    max_steps=TOTAL_STEPS,
    hidden_units=[8, 4],
    dropout_prob=0.0)

model_dir = 'trained_models/{}'.format(MODEL_NAME)

run_config = tf.estimator.RunConfig(
    save_checkpoints_steps=480, # to evaluate after each 20 epochs => (12000/500) * 20
    tf_random_seed=19830610,
    model_dir=model_dir
)

print("Model directory: {}".format(run_config.model_dir))
print("Hyper-parameters: {}".format(hparams))

#  #####定义服务函数，用于配置导出 #######
def csv_serving_input_fn()->tf.estimator.export.ServingInputReceiver:
    SERVING_HEADER = ['x', 'y', 'alpha', 'beta']
    SERVING_HEADER_DEFAULTS = [[0.0], [0.0], ['NA'], ['NA']]
    rows_string_tensor = tf.placeholder(dtype=tf.string, shape=[None], name='csv_rows')

    receiver_tensor = {'csv_rows': rows_string_tensor}

    row_columns = tf.expand_dims(rows_string_tensor, -1)
    columns = tf.decode_csv(row_columns, record_defaults=SERVING_HEADER_DEFAULTS)
    features = dict(zip(SERVING_HEADER, columns))

    if PROCESS_FEATURES:
        features = process_features(features)

    return tf.estimator.export.ServingInputReceiver(features, receiver_tensor)


# 定义训练和验证的输入函数
train_input_fn = lambda: csv_input_fn(
    file_name_pattern=TRAIN_DATA_FILES_PATTERN,
    mode=tf.estimator.ModeKeys.TRAIN,
    num_epochs=hparams.num_epochs,
    batch_size=hparams.batch_size
)

eval_input_fn = lambda: csv_input_fn(
    file_name_pattern=VALID_DATA_FILES_PATTERN,
    mode=tf.estimator.ModeKeys.EVAL,
    num_epochs=1,
    batch_size=hparams.batch_size)


train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,max_steps=hparams.max_steps,hooks=None)

eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
    exporters=[tf.estimator.LatestExporter(name="estimate",serving_input_receiver_fn=csv_serving_input_fn,exports_to_keep=1,as_text=True)],
    steps=None,
    throttle_secs = 15 # evalute after each 15 training seconds!
)



if RESUME_TRAINING:
    print("Removing previous artifacts...")
    shutil.rmtree(model_dir, ignore_errors=True)
else:
    print("Resuming training...")

tf.logging.set_verbosity(tf.logging.INFO)
time_start = datetime.utcnow()
print("Experiment started at {}".format(time_start.strftime("%H:%M:%S")))
print(".......................................")
estimator = create_estimator(run_config, hparams)

tf.estimator.train_and_evaluate(estimator=estimator, train_spec=train_spec, eval_spec=eval_spec)

time_end = datetime.utcnow()
print(".......................................")
print("Experiment finished at {}".format(time_end.strftime("%H:%M:%S")))
print("")
time_elapsed = time_end - time_start
print("Experiment elapsed time: {} seconds".format(time_elapsed.total_seconds()))
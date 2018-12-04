"""
@Name:        wrapper_estimator
@Description: ''
@Author:      Lucas Yu
@Created:     2018-11-17
@Copyright:   ©GYENNO Technology,Shenzhen,Guangdong 2018
@Licence:
"""

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from typing import List
import schema
from .model_selection import _safe_split
from .extract_transform_load import ConfigLoader


# 像spark的getOutputCol 或者 setInputCol,因为不是所有输出都是下一步想要的，scikit-learn借助于ColumnTransformer
class WrapperEstimator(BaseEstimator):

    def __init__(self, estimator=LogisticRegression(), conf=ConfigLoader.load())->None:
        self.estimator = estimator
        self.conf = conf

    def _feature_cols(self, data)->List[str]:
        data_cols = data.columns
        input_cols = schema.Schema([str]).validate(self.conf['FEATURE_COLS'].split())
        return [col for col in data_cols if col in input_cols]

    @property
    def _target_col(self)->str:
        return self.conf['TARGET_COL']

    @property
    def _weight_col(self)->str:
        return self.conf['WEIGHT']

    def fit(self, data, y=None):
        print(self.__class__.__name__, f'拟合估计器开始===================》》{data.columns}')
        features, target, weight = _safe_split(data, self._feature_cols(data), self._target_col, self._weight_col)
        self.estimator.fit(features, target)
        print(self.__class__.__name__, '拟合估计器结束==============')
        return self.estimator

    def predict(self, data):
        print(f"使用估计器预测！！！！！！！！！！！！！")
        features, target, weight = _safe_split(data, self._feature_cols(data), self._target_col, self._weight_col)
        return self.estimator.predict(features)

    def predict_proba(self,data):
        features, target, weight = _safe_split(data, self._feature_cols(data), self._target_col, self._weight_col)
        return self.estimator.predict_proba(features)

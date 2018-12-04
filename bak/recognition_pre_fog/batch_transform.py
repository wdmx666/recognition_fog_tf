"""
@Name:        md_test
@Description: ''
@Author:      Lucas Yu
@Created:     2018/11/5
@Copyright:   ©GYENNO Technology,Shenzhen,Guangdong 2018
@Licence:
"""
import pandas
import schema
from sklearn.pipeline import make_pipeline, Pipeline, FeatureUnion  # 二者是相对的，与ColumnTransformer不一样
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.validation import check_is_fitted
import itertools,functools
from typing import List
from joblib import Parallel,delayed

from .preprocessing import SignalStatusReMark, dynamic_scale_data, filter_data
from .feature_extraction import SignalFeatureExtractor, ExtractorManager
from .commons.common import JustTransformer
from .extract_transform_load import ConfigLoader

import warnings


# 必须了解到具体的内容的地方需要硬编码，知道变量类型等少量信息即可处理的地方可以避免
class BatchTransformer(JustTransformer):
    def __init__(self, conf=ConfigLoader.load(), groupby='sample_name'):
        self.conf = conf
        self.groupby: str = schema.Schema(str).validate(groupby)
        schema.Use(lambda x: warnings.warn('最好使用样本名分组') if x != "sample_name" else None).validate(groupby)
        self.BIG_PIPE = self._transformer

    @property
    def _feature_names(self)->List[str]:
        return schema.Schema([str]).validate(self.conf['FEATURE_NAMES'])

    @property
    def _signal_cols(self)->List[str]:
        return schema.Schema([str]).validate(self.conf['SIGNAL_COLS'])

    @property
    def _no_signal_cols(self)->List[str]:
        return schema.Schema([str]).validate(self.conf['NO_SIGNAL_COLS'])

    @property
    def _feature_manager(self):
        return ExtractorManager().make_extractors()

    @property
    def _transformer(self):  # 该过程根据实际过程很特异，重用可能性不大
        ssr = SignalStatusReMark()
        fu = FeatureUnion([(it, SignalFeatureExtractor(self._feature_manager.get_extractor(it))) for it in self._feature_names])

        filter_trans = FunctionTransformer(filter_data)
        scale_trans = FunctionTransformer(dynamic_scale_data)

        num_pipe = Pipeline([('fu', fu), ('filter_trans', filter_trans), ('scale_trans', scale_trans)])

        ct = ColumnTransformer([('pipe', num_pipe, self._signal_cols),
                                ("Identity", SignalFeatureExtractor(self._feature_manager.get_extractor("Identity")), self._no_signal_cols)])
        BIG_PIPE = make_pipeline(ssr, ct)
        return BIG_PIPE

    def _fit_transform(self, x, y=None, fitted=False):
        fu_gen_cols = ["_".join(i) for i in itertools.product(self._signal_cols, self._feature_names)]
        groups = x[self.groupby].unique()
        result_df = pandas.DataFrame()

        def fit_one(data, fitted_or_not):
            if fitted_or_not:
                result = self.BIG_PIPE_.transform(data)
            else:
                result = self.BIG_PIPE.fit_transform(data)
            return result

        for group in groups:
            df4cal = x[x[self.groupby] == group]
            res = fit_one(df4cal, fitted)
            df = pandas.DataFrame(data=res, columns=fu_gen_cols + self._no_signal_cols)
            result_df = result_df.append(df, ignore_index=True)
        self.BIG_PIPE_ = self.BIG_PIPE

        # parallel = Parallel(n_jobs=3)
        # result_df = pandas.DataFrame()
        # with parallel:
        #     results = parallel(delayed(self._transform)(x[x[self.groupby] == group]) for group in groups)

        return result_df

    def transform(self, x, y=None, **fit_params):
        if not hasattr(self, "BIG_PIPE_"):
            print("没有fit")
            raise NotImplementedError
        result_df = self._fit_transform(x, fitted=True)

        return result_df

    def fit_transform(self, x: pandas.DataFrame, y=None, **fit_params)->pandas.DataFrame:
        # 输入测试集时候不应执行fit，前面的transformer必须fitted并且转transform出数据，才能让后续的transformer进行fitted
        print(self.__class__.__name__, "=======fit_transform 开始==============")
        result_df = self._fit_transform(x)
        print(self.__class__.__name__, "=======fit_transform 结束==============")
        return result_df



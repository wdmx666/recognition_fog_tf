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
    def _transformer(self):
        ssr = SignalStatusReMark()
        fu = FeatureUnion([(it, SignalFeatureExtractor(self._feature_manager.get_extractor(it))) for it in self._feature_names])

        filter_trans = FunctionTransformer(filter_data)
        scale_trans = FunctionTransformer(dynamic_scale_data)

        num_pipe = Pipeline([('fu', fu), ('filter_trans', filter_trans), ('scale_trans', scale_trans)])

        ct = ColumnTransformer([('pipe', num_pipe, self._signal_cols),
                                ("Identity", SignalFeatureExtractor(self._feature_manager.get_extractor("Identity")), self._no_signal_cols)])
        BIG_PIPE = make_pipeline(ssr, ct)
        return BIG_PIPE

    def fit(self, x, y=None, **fit_params):
        # 输入测试集时候不应执行fit，前面的transformer必须fitted并且转transform出数据，才能让后续的transformer进行fitted
        print(self.__class__.__name__, ">>>>>>>被拟合>>>>>>>>>>>>>>", x['sample_name'][0].tolist())
        pipe = self._transformer
        groups = x[self.groupby].unique()
        for group in groups:
            df4cal = x[x[self.groupby] == group]
            pipe.fit(df4cal, y)
        self.BIG_PIPE_ = pipe
        print(self.__class__.__name__, ">>>>>>>拟合结束>>>>>>>>>>>>>>")
        return self

    def transform(self, x: pandas.DataFrame)->pandas.DataFrame:
        print(self.__class__.__name__, "=======转换开始==============")
        fu_gen_cols = ["_".join(i) for i in itertools.product(self._signal_cols, self._feature_names)]
        parallel = Parallel(n_jobs=3)
        groups = x[self.groupby].unique()
        result_df = pandas.DataFrame()

        # with parallel:
        #     results = parallel(delayed(self._transform)(x[x[self.groupby] == group]) for group in groups)
        for group in groups:
            df4cal = x[x[self.groupby] == group]
            res = self.BIG_PIPE_.transform(df4cal)
            df = pandas.DataFrame(data=res, columns=fu_gen_cols + self._no_signal_cols)
            result_df = result_df.append(df, ignore_index=True)
        # for df in results:
        #     result_df = result_df.append(df)
        print(self.__class__.__name__, "=======转换结束==============")
        return result_df



"""
@Name:        signal_mark_transform
@Description: ''
@Author:      Lucas Yu
@Created:     2018/11/12
@Copyright:   (c) GYENNO Science,Shenzhen,Guangdong 2018
@Licence:
"""
import numpy
from sklearn import preprocessing
import functools
from pandas import DataFrame, Interval
from typing import Dict, List,Union
from schema import And, Or, Use, Schema
from ..extract_transform_load import ConfigLoader

from ..commons.common import JustTransformer,NamedDataFrame
from ..commons.utils import NamedDataFrame2CSV


# TODO(yu): 不能采用标准的变换器吗,FunctionTransformer不能带参数,只能向其传递纯函数，
# TODO(yu): 通过ColumnTransformer可指定(name, transformer, column(s)),可指定转换器作用列，
# TODO(yu): 转换器参数还是只能转换器自带，也即自造转换器；通常定义的转换器多是对传入的数据的所有列按列单独操作。
class SignalStatusReMark(JustTransformer):
    """全字段操作，不能与下面选择字段连接
    """
    def __init__(self, conf=ConfigLoader, pre_fog_time_len=200):
        self.conf = conf
        self.pre_fog_time_len: Union[int, float] = Schema(And(Use(float), Or(int, float))).validate(pre_fog_time_len)

    @property
    def _status_definition(self):
        return self.conf.load('doc/para.conf')['STATUS_DEFINITION']

    def _mark_pre_fog_according_to_fog(self, df: DataFrame)->DataFrame:
        df.loc[:, 're_status'] = df['status']

        def get_change_point(dfs):  # 使用二元窗找到切换点
            status_change = []
            for idx in dfs.head(dfs.shape[0] - 1).index:
                if dfs.loc[idx, 'status'] != 'fog' and dfs.loc[idx + 1, 'status'] == 'fog':
                    status_change.append(idx)
            return status_change

        # 根据切换点和定义pre_fog时间长标记
        def re_status(dfs, change_point, time_len)->DataFrame:
            con = dfs.loc[change_point, 'time10'] - time_len
            for i in range(change_point, -1, -1):
                if dfs.loc[i, 'time10'] < con or dfs.loc[i, 'status'] == 'fog':
                    break
                else:
                    dfs.loc[i, 're_status'] = 'pre_fog'
            return dfs
        status_changes = get_change_point(df)
        for it in status_changes:
            df = re_status(df, it, self.pre_fog_time_len)
        return df

    @staticmethod
    def _mark_status(it, status_definition, status="normal"):
        for k, lst in status_definition.items():
            if it in lst:
                status = k
                break
        return status

    # @NamedDataFrame2CSV(save_or_not=True)
    def transform(self, df: DataFrame):  # mark the "pre_fog" status according to the "fog"
        print(self.__class__.__name__, "转换开始", )
        df.loc[:, 'status'] = df['gait_type'].apply(self._mark_status, args=(self._status_definition,))
        result_df = self._mark_pre_fog_according_to_fog(df)
        print(self.__class__.__name__, "转换结束", list(result_df.columns[-11:]))
        return result_df


# 在sklearn中用ColumnTransformer来指定操作的列，这与spark的提取器本身维护输入输出列不同
# 检查参数的赋值是否规范，是否建立一个参数管理器，规定参数的定义范围，检查输入的参数是否符合定义域
# 谁直接使用参数谁负责检查
class DataSelector(JustTransformer):
    def __init__(self, conf=ConfigLoader.load()):
        self.conf = conf  # 配置器执行输入字段检查,类型检查太费事

    @property
    def _select_condition(self)->List[str]:
        return Schema([str]).validate(self.conf["STATUS_VALUES"])

    @property
    def _target_col(self)->str:
        return Schema(str).validate(self.conf["TARGET_COL"])

    @property
    def _weight_cols(self):
        return Schema([str]).validate(self.conf["WEIGHT_COLS"].split())

    def transform(self, x: DataFrame)->DataFrame:
        print(self.__class__.__name__, "===选择数据转换开始===》》")
        from ._base import add_weight
        print("选择数据", x.columns[0:3])
        dfs = x[x[self._target_col].isin(self._select_condition)]
        dfs.loc[:, 'weight'] = add_weight(dfs[self._weight_cols].values)
        print(self.__class__.__name__, "===选择数据转换结束===》》")
        return dfs







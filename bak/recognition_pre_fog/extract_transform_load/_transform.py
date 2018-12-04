"""
@Name:        transformers
@Description: 准备数据样本
@Author:      Lucas Yu
@Created:     2018/11/5
@Copyright:   ©GYENNO Technology,Shenzhen,Guangdong 2018
@Licence:
"""
import pandas
from pandas import DataFrame
import numpy
import copy
import path
import sys
import itertools
import keras
from schema import Schema
from typing import List, Dict

from ..commons.common import JustTransformer,NamedDataFrame
from ..commons.utils import NamedDataFrame2CSV

from .preparation import ConfigLoader


# 涉及多列之间的交互，每列之间不可能独立计算，不同列参与的方式不一样，所以仍需列名
class DropDuplicateFillLostTimePoint(JustTransformer):
    @NamedDataFrame2CSV(save_or_not=True)
    def transform(self, named_df: NamedDataFrame)->NamedDataFrame:
        df = named_df.dataframe
        df = df.drop_duplicates(subset=['time10'])
        df = df.reset_index(drop=True, )
        lost_row = []
        dtf = df["time10"].diff().dropna()
        dtf = dtf[dtf > 1]
        for idx, v in dtf.iteritems():
            avg = pandas.Series.add(df.loc[idx], df.loc[idx - 1]) / 2
            time_range = range(df["time10"].loc[idx - 1], df["time10"].loc[idx], 1)
            for i in range(1, len(time_range)):
                ds = copy.deepcopy(avg)
                ds["time10"] = int(time_range[i])
                lost_row.append(ds.astype(int))
                # print(time_range[i])
        result_df = df.append(pandas.DataFrame(lost_row), ignore_index=True).sort_values("time10").reset_index(drop=True)
        return NamedDataFrame(named_df.name, result_df)


# todo: 数据内容检查？
class SignalMarker(JustTransformer):
    """将这些关系紧密的代码最好不分开为妙"""
    def __init__(self, maker_path: str = '../data/VideoMarkDataExtractor', status_definition=None)->None:
        """因为程序的正确执行不但与输入数据的类型有关，还与数据的字段有关，所以类型安全并不能保证程序可控，
        程序可能因内容不安全，因此检查较难"""
        self.marker_path = path.Path(maker_path)
        tmp = ConfigLoader.load('doc/para.conf')['STATUS_DEFINITION'] if not status_definition else status_definition
        self.status_definition: Dict[str, List[str]] = Schema({str: [str]}).validate(tmp)

    def _check_parameters(self):
        if not self.marker_path.exists():
            raise FileNotFoundError(f"请正确指明，依赖目录")


    @staticmethod
    def _mark_signal_with_video(df_s: DataFrame, df_v: DataFrame)->DataFrame:
        df_all = df_s.reindex(columns=list(df_s.columns) + list(df_v.columns))
        for i in df_v.index:
            v_it = df_v.loc[i, :]
            start = v_it['start_time']
            end = v_it['end_time']
            idx = df_all[(df_all['time10'] >= start) & (df_all['time10'] <= end)].index.tolist()
            df_all.loc[idx, v_it.index.tolist()] = v_it.tolist()
        return df_all

    @staticmethod
    def _drop_head_tail_na(df_all: DataFrame, col_name: str)->DataFrame:
        start_id = 0
        for i in df_all.index:
            if pandas.notna(df_all.loc[i, col_name]):
                start_id = i
                break
        end_id = len(df_all)
        for j in df_all.index[::-1]:
            if pandas.notna(df_all.loc[j, col_name]):
                end_id = j
                break
        print(start_id, end_id)
        return df_all.loc[start_id:end_id, :]

    @staticmethod
    def _mark_status(it, status_definition: Dict[str, List[str]], status="normal")->str:
        for k, lst in status_definition.items():
            if it in lst:
                status = k
                break
        return status

    @NamedDataFrame2CSV(save_or_not=True)
    def transform(self, signal_named_df: NamedDataFrame)->NamedDataFrame:
        """the low layer(the general layer) should not know much about the above(the business)"""
        s_df_name: str = signal_named_df.name
        df_s: DataFrame = signal_named_df.dataframe
        pid = path.Path(s_df_name).namebase.split("==")[0]
        try:
            pair_name = next(filter(lambda it: it.find(pid) > -1, self.marker_path.files()))
            df_v = pandas.read_csv(open(pair_name))
        except (FileNotFoundError, StopIteration) as e:
            print(f"文件不存在，请先获取依赖文件,因为 {e} 退出！")
            sys.exit(1)
        print(df_v.columns, df_v.head(2))
        df_all = self._mark_signal_with_video(df_s, df_v)
        df_res = self._drop_head_tail_na(df_all, 'end_time')
        gait_type = list(itertools.chain.from_iterable(self.status_definition.values()))
        # 将不再已知列表内的步态全部标记为N
        df_res.loc[:, 'gait_type'] = df_res['gait_type'].apply(lambda it: it if it in gait_type else 'N')
        # 根据步态类型所对应的状态，将status标记为fog与normal
        df_res.loc[:, 'status'] = df_res['gait_type'].apply(self._mark_status, args=(self.status_definition,))
        # 根据status,将其编码为数字
        df_res.loc[:, 'status_code'] = df_res.loc[:, 'status'].apply(lambda it: 1 if it == 'fog' else 0)
        # 根据状态码将其标记为独热码
        return NamedDataFrame(s_df_name, df_res)



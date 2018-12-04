import numpy as np
import pandas as pd
import path
from collections import Counter
from ..commons.common import MyCalculator,MyProperties
from imblearn.over_sampling import SMOTE, ADASYN
from ..client.proj_config import memory
import numba




class StrategyResult2(object):
    def __init__(self):
        pass

    def mark_result(self, df, time_len_filter=200,proba=0.5,time_len_event=500):
        """将时间列选择出来，以时间窗口的形式单步滑窗滤波，原始真值不参与滤波"""
        # 将真实状态用01表示，normal为1
        df.loc[:, "status01"] = df["re_status"].apply(lambda x: 0 if x == "normal" else 1)
        # 以当前时间点为起始，选出窗口长度为time_len_filter的数据段进行中值滤波
        # 将使用窗口中值作为本时间点的真值估计并结合概率阈值判断预测的状态
        fm = FilterMarker(proba)
        filtered_predict_status_01 = [fm.transform(df["predict_status_proba2"][(df["time10"] > time_point - time_len_filter) &
                                                                              (df["time10"] <= time_point)]) for time_point in df["time10"]]
        df.loc[:, "filtered_predict_status01"] = filtered_predict_status_01
        # 一阶差分
        df.loc[:, "dt"] = df["time10"].diff(1).tolist()
        df.loc[:, "dfps"] = df["status01"].diff(1).tolist()
        df.loc[:, "filtered_predict_dfps"] = df["filtered_predict_status01"].diff(1).tolist()
        df = df.dropna()
        et = EventCounter(time_len_event, 0)
        df["event_id"] = df[["dt", "dfps"]].apply(lambda x: et.transform(x[0], x[1]), axis=1)
        et = EventCounter(time_len_event, 0)
        df["filtered_predict_event_id"] = df[["dt", "filtered_predict_dfps"]].apply(lambda x: et.transform(x[0], x[1]), axis=1)
        return df


class EventCounter:
    def __init__(self, time_len, counter=0):
        self.counter = counter
        self.time_len = time_len

    def transform(self, x1, x2):
        if x1 < self.time_len and x2 == 0:  # 只有这种情况下才是同一个事件，其余三种情况都是新事件产生
            return self.counter
        else:
            self.counter += 1
            return self.counter


class FilterMarker:
    def __init__(self, proba):
        self.proba_threshold = proba

    def transform(self, series):
        return 1 if np.median(series) > self.proba_threshold else 0


class MarkRules(object):
    """具体的滤波和事件标记规则"""
    @staticmethod
    def mark_rule_proba(series, proba=0.5):
        if np.median(series) > proba:
            return 1
        else:
            return 0

    @staticmethod
    def mark_rule_status(series):
        real_ct = Counter(series)
        if real_ct["pre_fog"] / len(series) > 0.1:
            return 1
        else:
            return 0

    @staticmethod
    def mark_event(df, time_len=500):
        # mark the event
        """
        如果时间差和状态没发生改变则将该事件标记为同一个；
        时间差值大于规定时间长度则分为两个事件；
        若状态发生改变则标记为新事件"""
        df.loc[:, "dt"] = df["time10"].diff(1).tolist()
        df.loc[:, "dfps"] = df["filtered_predict_status"].diff(1).tolist()
        df = df.dropna()
        c = 0
        E = []
        for i in df.index:
            if df.loc[i, "dt"] < time_len and df.loc[i, "dfps"] == 0:  # 只有这种情况下才是同一个事件，其余三种情况都是新事件产生
                E.append(c)
            else:
                c += 1
                E.append(c)
        df.loc[:, "event_id"] = E
        return df


# 这个模型的历史统计，包括两种状态（preFoG和normal），只是为了说明当实际情况是preFoG和normal时它有多好
class Utils:

    @classmethod
    def evaluation(cls, df):
        conf_mat = {0: {0: 0, 1: 0},
                    1: {0: 0, 1: 0}}
        predicted_time = []
        dfg = df.groupby("event_id").first()  # 根据预测事件的ID来分组，并且分组之后取得组内第一个属性值
        ls = len(dfg.index)
        for event_id in range(ls):  # 迭代事件ID,选出每个预测事件的数据
            if dfg.index[event_id] < dfg.index[-1]:
                dfe = df[(df["time10"] >= dfg.loc[dfg.index[event_id], "time10"]) & (
                       df["time10"] < dfg.loc[dfg.index[event_id + 1], "time10"])]
            else:
                dfe = df[(df["time10"] >= dfg.loc[dfg.index[event_id], "time10"])]
            # 统计预测事件时间段内真实状态的计数
            ct = Counter(dfe["filtered_predict_status01"]) # 统计预测事件中01的数量，后面根据有无1判断是否有交集来判断是否命中
            ds = dfg.loc[dfg.index[event_id], "status01"]   # 根据索引找到当前事件的第一个值作为真实事件的类型
            if ds == 0:   # 当实际事件为0类型时进行统计
                if ct[0] / (ct[0] + ct[1] + 0.1) > 0.8:  # 当分割内0事件的比例超过阈值时则认为该分割与预测事件一致，00+1
                    conf_mat[0][0] += 1  # 真阴性
                else:
                    conf_mat[0][1] += 1  # 假阳性
            elif ds == 1:  # 当真实事件为1类型时进行统计,又交集即命中
                if ct[1] >= 1:  # 预测事件
                    conf_mat[1][1] += 1  # 实际1预测1，命中
                    # 使用dfe(真实事件)选择预测事件为1的数据
                    flag = dfe[dfe["filtered_predict_status01"] == 1]["filtered_predict_event_id"].iloc[0]
                    start_time = df[df["filtered_predict_event_id"] == flag]["time10"].iloc[0]
                    # end_time定义为dfe的最后一个值
                    end_time = dfe["time10"].iloc[-1]
                    predicted_time.append(end_time - start_time)
                else:
                    conf_mat[1][0] += 1  # 实际1预测0，漏报
                    predicted_time.append(0)  # 提前时间为0
            else:
                print("fault")

        res = pd.DataFrame([(0, 0)] * conf_mat[0][0] + [(0, 1)] * conf_mat[0][1] +
                           [(1, 0)] * conf_mat[1][0] + [(1, 1)] * conf_mat[1][1], columns=["truth","predict"])

        print("预测结果！=========", res, "=====================")

        return res, predicted_time

    @classmethod
    def evaluation_v1_1(cls, df):
        conf_mat = {0: {0: 0, 1: 0},
                    1: {0: 0, 1: 0}}
        predicted_time = []
        print(df.head())
        dfg = df.groupby("filtered_predict_event_id").first()  # 根据预测事件的ID来分组，并且分组之后取得组内第一个属性值
        # print(dfg)
        ls = len(dfg.index)
        for predict_event_id in range(ls):  # 迭代事件ID,选出每个预测事件的数据
            if dfg.index[predict_event_id] < dfg.index[-1]:
                dfe = df[(df["time10"] >= dfg.loc[dfg.index[predict_event_id], "time10"]) & (
                       df["time10"] < dfg.loc[dfg.index[predict_event_id + 1], "time10"])]
            else:
                dfe = df[(df["time10"] >= dfg.loc[dfg.index[predict_event_id], "time10"])]
            # 统计预测事件时间段内真实状态的计数
            ct = Counter(dfe["status01"])  # 对真实事件进行了分割(以预测事件)，这样对统计混淆矩阵有用
            # 根据预测事件的顺序获取对应的索引，根据索引找到当前事件的第一个值作为预测事件的类型
            ds = dfg.loc[dfg.index[predict_event_id], "filtered_predict_status01"]
            if ds == 0:   # 当预测事件为0类型时进行统计
                if ct[0] / (ct[0] + ct[1] + 0.1) > 0.8:  # 当分割内0事件的比例超过阈值时则认为该分割与预测事件一致，00+1
                    conf_mat[0][0] += 1
                else:
                    conf_mat[0][1] += 1
                    predicted_time.append(0)
            elif ds == 1:  # 当预测事件为1类型时进行统计
                if ct[1] >= 1:  # 实际事件
                    conf_mat[1][1] += 1
                    # 使用dfe(预测事件)选择真实事件为1的数据
                    dfe_s = dfe[dfe["status01"] == 1]
                    # 初始化end_time为dfe_s的第一个值
                    end_time = dfe_s["time10"].iloc[0]
                    # start_time定义为dfe的第一个值
                    start_time = dfe["time10"].iloc[0]
                    # print(dfe_s)
                    # 选择出endtime之后的数据的index
                    df_s4et = df[df["time10"] >= end_time].index
                    # 选取真实的事件的初始值ID,若ID不变继续往下迭代，变了则跳出更新end_time
                    flag = dfe_s["event_id"].iloc[0]
                    for i in df_s4et:
                        if df["event_id"].loc[i] == flag:
                            # print(df["filtered_status"].loc[i],"break!")
                            end_time = df["time10"].loc[i]
                        else:
                            break
                    predicted_time.append(end_time - start_time)
                else:
                    conf_mat[1][0] += 1
            else:
                print("fault")

        res = pd.DataFrame([(0, 0)] * conf_mat[0][0] + [(0, 1)] * conf_mat[0][1] +
                           [(1, 0)] * conf_mat[1][0] + [(1, 1)] * conf_mat[1][1], columns=["predict", "truth"])

        return res, predicted_time

    @classmethod
    def evaluation_v1(cls, df):
        conf_mat = {0: {0: 0, 1: 0},
                    1: {0: 0, 1: 0}}
        predicted_time = []
        dfg = df.groupby("event_id").first()
        # print(dfg)
        ls = len(dfg.index)
        for i in range(ls):
            if dfg.index[i] < dfg.index[-1]:
                dfe = df[(df["time10"] >= dfg.loc[dfg.index[i], "time10"]) & (
                df["time10"] < dfg.loc[dfg.index[i + 1], "time10"])]
            else:
                dfe = df[(df["time10"] >= dfg.loc[dfg.index[i], "time10"])]
            ct = Counter(dfe["status01"])
            # print(dfe.shape,ct)
            ds = dfg.loc[dfg.index[i], "filtered_predict_status01"]
            if ds == 0:
                if ct[0] / (ct[0] + ct[1] + 0.1) > 0.8:
                    conf_mat[0][0] += 1
                else:
                    conf_mat[0][1] += 1
                    predicted_time.append(0)
            elif ds == 1:
                if ct[1] >= 1:
                    conf_mat[1][1] += 1
                    dfe_s = dfe[dfe["status"] == 1]

                    end_time = dfe_s["time10"].iloc[0]
                    start_time = dfe["time10"].iloc[0]
                    # print(dfe_s)
                    df_s4et = df[df["time10"] >= end_time].index
                    for i in df_s4et:
                        if (df["filtered_status"].loc[i] == 0) or (df["dt"].loc[i]>200):
                            # print(df["filtered_status"].loc[i],"break!")
                            break
                        end_time = df["time10"].loc[i]
                    predicted_time.append(end_time - start_time)
                else:
                    conf_mat[1][0] += 1
            else:
                print("fault")

        res = pd.DataFrame([(0, 0)] * conf_mat[0][0] + [(0, 1)] * conf_mat[0][1] +
                           [(1, 0)] * conf_mat[1][0] + [(1, 1)] * conf_mat[1][1], columns=["predict", "truth"])

        return res, predicted_time





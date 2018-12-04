"""
@Name:        validation
@Description: ''
@Author:      Lucas Yu
@Created:     2018/11/15
@Copyright:   (c) GYENNO Science,Shenzhen,Guangdong 2018
@Licence:
"""

"""模型验证基本步骤：(1) 选择评价指标，(2) 选择评价方式，(3) 确定的模型
   使用模型和数据计算处评价指标，在选定的评价方式下多次在多个样本上进行，
   从而得到客观评价
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import path
import collections
import os
from sklearn import metrics

os.environ['R_USER'] = r'D:\ProgramLanguageCore\Python\anaconda351\Lib\site-packages\rpy2'
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from .algo_core import Utils
from ..commons.common import MyCalculator
import numba


# 单次验证
def _fit_and_score(estim)

class FitAndScore(MyCalculator):
    def __init__(self, name=None):
        super(TrainTestSplitValidator, self).__init__(name)
        self.para.setdefault("data_maker", None)
        self.para.setdefault("model", None)
        self.para.setdefault("strategy", None)


    def calculate(self, msg):
        """针对每一个患者样本,采用留一法训练样本和测试样本，返回样本预测结果包括状态和其概率"""
        df_train = self.para["data_maker"].prepare_train_df(msg[1])
        df_val = self.para["data_maker"].prepare_val_df(msg[0])
        cols = self.para["data_maker"].out_cols
        clf = self.para["model"]

        print(self.name, "数据准备完毕，进行拟合及评价")
        clf.fit(df_train[cols['feature_cols']], df_train[cols['target_cols']].values.reshape(-1), sample_weight=df_train[cols['weight_cols']].values.reshape(-1))

        result_df = df_val[["time10"] + cols['target_cols']]
        result_df.loc[:, "predict_status"] = clf.predict(df_val[cols['feature_cols']])
        proba = clf.predict_proba(df_val[cols['feature_cols']])

        result_df.loc[:, "predict_status_proba1"] = [i[0] for i in proba]
        result_df.loc[:, "predict_status_proba2"] = [i[1] for i in proba]

        msg = {"data": result_df, "filename": path.Path(msg[0][0])}  # 将作为验证的文件继续文件名记下通过返回值往下传递
        result_df = self.para["strategy"].mark_result(msg["data"])
        print(metrics.classification_report(result_df["status01"], result_df["filtered_predict_status01"]))

        result = dict()
        result["raw_f1_score"] = metrics.f1_score(result_df[cols['target_cols'][0]], result_df["predict_status"], average='micro')
        result["filtered_f1_score"] = metrics.f1_score(result_df["status01"],result_df["filtered_predict_status01"], average='micro')
        result["raw_kappa"] = metrics.cohen_kappa_score(result_df[cols['target_cols'][0]], result_df["predict_status"])
        result["filtered_kappa"] = metrics.cohen_kappa_score(result_df["status01"], result_df["filtered_predict_status01"])
        result["train_score"] = clf.score(df_train[cols['feature_cols']], df_train[cols['target_cols']].values.reshape(-1),
                                          sample_weight=df_train[cols['weight_cols']].values.reshape(-1))
        result["validation_score"] = clf.score(df_val[cols['feature_cols']], df_val[cols['target_cols']].values.reshape(-1),
                                               sample_weight=df_val[cols['weight_cols']].values.reshape(-1))
        result["predict_result"] = result_df


class CVProtocol(object):

    def split(self, samples_dir):
        return self.__leave_one_patient_out(samples_dir)

    def __leave_one_patient_out(self, samples_dir):
        p = path.Path(samples_dir)
        print("sample_dir", samples_dir)
        f4model = []
        fp = np.array([p.joinpath(i) for i in p.files()])
        for i in range(len(fp)):
            fpr = np.roll(fp, -i)
            f4model.append(([fpr[0]], list(fpr[1:])))
        return f4model


class CrossValidator(MyCalculator):
    def __init__(self, name=None):
        super().__init__(name)
        self.para.setdefault("TrainTestSplitValidator", None)
        self.para.setdefault("CVProtocol", None)

    @numba.jit
    def calculate(self, samples_dir):
        result = dict()  # one request result
        items = collections.deque(self.para["CVProtocol"].split(samples_dir))
        #print(self.name, items)
        # 注意items的每一项item都是列表的元组，即item的每项都是列表

        while items:
            item = items.popleft()
            print(self.name, item)
            result[path.Path(item[0][0]).basename().__str__()] = self.para['TrainTestSplitValidator'].calculate(item)
            print('<<<<==========>>>>')
        #print("one_cv_result: ", result)
        return result


class EventParaValidator(MyCalculator):
    def __init__(self, name=None):
        super().__init__(name)
        self.para.setdefault("strategy", None)
        self.para.setdefault("data_path", None)
        self.para.setdefault("save_path", None)

    def set_para_with_prop(self, my_props):
        self.para.update(my_props)

    def calculate(self, msg):
        c = 0
        ID, para = msg
        que = collections.deque(path.Path(self.para['data_path']).files())
        all_predicted_time = dict()
        rs = pd.DataFrame()
        print("que：", len(que))
        while que:
            item = que.popleft()
            df_tmp = pd.read_csv(open(item))
            mark_result = self.para['strategy'].mark_result(df_tmp, para["filter_time"], para["probability_value"],para["event_time"])
            res_all = Utils.evaluation(mark_result)
            print(res_all)
            res = res_all[0]
            predicted_time = res_all[1]
            f1 = metrics.f1_score(res["truth"], res["predict"], average="macro")
            c += 1
            res["sample_name"] = path.Path(item).basename()
            rs = rs.append(res)
            all_predicted_time.update({path.Path(item).basename(): predicted_time})
            print(rs.columns,"=============================================the %d" % c)

        #print(all_predicted_time)
        save_path = path.Path(self.para["save_path"]).joinpath("event_para_"+ID)
        #input_data = all_predicted_time.values()
        #print(metrics.classification_report(rs["truth"], rs["predict"]))
        print(rs.head())
        #print(metrics.f1_score(rs["truth"], rs["predict"], average="macro"), metrics.cohen_kappa_score(rs["truth"], rs["predict"]))
        EventPlotter().calculate((save_path, [[j for j in i if j is not None] for i in all_predicted_time.values()]))  # 偷了一个懒，没有注入
        return all_predicted_time, rs
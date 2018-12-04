"""
@Name:        features_etl
@Description: 将特征当作下步骤的原始数据，进行数据的ETL，也包括弥补确实去离群还有标准化，与上步成递归关系，
              背景的意义不同但所执行的动作类型相同，也就是说这些动作是通用的，与特定环境类型无关的。
@Author:      Lucas Yu
@Created:     2018/11/6
@Copyright:   (c) GYENNO Science,Shenzhen,Guangdong 2018
@Licence:
"""

import numpy as np
import abc
from scipy import stats,fftpack
import numba



# 特征对照列表
# (最大值，最小值，绝对平均值，峰峰值，均方根，均值，标准差，偏态，峭度，方差，
#  波形因子，波峰因子，变异系数，偏态系数，余隙因子，脉冲因子，能量因子，
#  平均频率，中心频率，频率均方根，频率标准差)
features_info={"F01":"最大值","F02":"最小值","F03":"绝对平均值","F04":"峰峰值",
              "F05":"均方根","F06":"均值","F07":"标准差","F08":"偏态",
              "F09":"峭度","F10":"方差","F11":"波形因子","F12":"波峰因子",
              "F13":"变异系数","F14":"偏态系数","F15":"余隙因子","F16":"脉冲因子",
              "F17":"脉冲因子","F18":"平均频率","F19":"平均频率","F20":"中心频率",
              "F21":"频率均方根","F22":"频率标准差"}


# 接口/抽象类4定义各种具体的特征提取器
class FeatureExtractor(metaclass=abc.ABCMeta):
    def run(self, series):
        pass


# 时域特征计算 ##
# (最大值，最小值，绝对平均值，峰峰值，均方根，均值，标准差，偏态，峭度，方差，
#  波形因子，波峰因子，变异系数，偏态系数，余隙因子，脉冲因子，能量因子，
# 所有的具体特征提取器都是无状态的，不会进行状态维护 #


# 为什么采用类的方式而不是函数来定义特征提取器，主要是有的方法是带参数的。
class Identity(FeatureExtractor):
    """对于有限长度信号序列时域幅值，求最大值
    """
    @numba.jit(cache=True,parallel=True)
    def run(self, series):
        return series[-1]

class F1(FeatureExtractor):
    """对于有限长度信号序列时域幅值，求最大值
    """
    def __init__(self):
        self.c=0

    @numba.jit(cache=True,parallel=True)
    def run(self, series):
        self.c += 1
        #series
        #print(self.c,"-->",series[0:5])
        return round(np.max(series), 2)


class F2(FeatureExtractor):
    """对于有限长度信号序列时域幅值，求最小值
    """
    @numba.jit(cache=True,parallel=True)
    def run(self, series):
        return round(np.min(series), 2)


class F3(FeatureExtractor):
    """对于有限长度信号序列时域幅值，先求绝对值，再求均值
    """
    @numba.jit(cache=True,parallel=True)
    def run(self, series):
        return round(np.mean(np.abs(series)), 2)


class F4(FeatureExtractor):
    """对于有限长度信号序列时域幅值，求极差
    """
    @numba.jit(cache=True,parallel=True)
    def run(self, series):
        return round(np.max(series) - np.min(series), 2)


class F5(FeatureExtractor):
    """对于有限长度信号序列时域幅值，求均方根--平方后求平均开根号
    """
    @numba.jit(cache=True,parallel=True)
    def run(self, series):
        return round(np.sqrt(np.mean(np.square(series))), 2)


class F6(FeatureExtractor):
    """对于有限长度信号序列时域幅值，求均值
    """
    @numba.jit(cache=True,parallel=True)
    def run(self, series):
        return round(np.mean(series), 2)


class F7(FeatureExtractor):
    """对于有限长度信号序列时域幅值，求标准差
    """
    @numba.jit(cache=True,parallel=True)
    def run(self, series):
        return round(np.std(series), 2)


class F8(FeatureExtractor):
    """对于有限长度信号序列时域幅值，求偏度（标准化三阶中心矩）
       三阶中心矩/三阶标准差
    """
    @numba.jit(cache=True,parallel=True)
    def run(self, series):
        return round(stats.moment(series, 3) / np.power(np.std(series), 3), 2)


class F9(FeatureExtractor):
    """对于有限长度信号序列时域幅值，求偏度（标准化四阶中心矩）
       四阶中心矩/四阶标准差
    """
    @numba.jit(cache=True,parallel=True)
    def run(self, series):
        return round(stats.moment(series, 4) / np.power(np.std(series), 4), 2)


class F10(FeatureExtractor):
    """对于有限长度信号序列时域幅值，求方差(二阶中心矩))
    """
    @numba.jit(cache=True,parallel=True)
    def run(self, series):
        return round(np.var(series), 2)


class F11(FeatureExtractor):
    """对于有限长度信号序列时域幅值，求波形因子--绝对平方均值开根号/绝对均值
    """
    @numba.jit(cache=True,parallel=True)
    def run(self, series):
        return round(np.sqrt(np.mean(np.square(np.abs(series)))) / np.mean(np.abs(series)), 2)


class F12(FeatureExtractor):
    """对于有限长度信号序列时域幅值，求波峰因子--最大值/平方均值开根号
    """
    @numba.jit(cache=True,parallel=True)
    def run(self, series):
        return round(np.max(series) / np.sqrt(np.mean(np.square(series))), 2)


class F13(FeatureExtractor):
    """对于有限长度信号序列时域幅值，求变异系数--均值/标准差
    """
    @numba.jit(cache=True,parallel=True)
    def run(self, series):
        return round(np.mean(series) / np.std(series), 2)


class F14(FeatureExtractor):
    """对于有限长度信号序列时域幅值，求偏态系数--信号/标准差之后三次方求均值
    """
    @numba.jit(cache=True,parallel=True)
    def run(self, series):
        return round(np.mean(np.power(series / np.std(series), 3)), 2)


class F15(FeatureExtractor):
    """对于有限长度信号序列时域幅值，求峭度系数--信号/标准差之后四次方求均值
    """

    @numba.jit(cache=True,parallel=True)
    def run(self, series):
        return round(np.mean(np.power(series / np.std(series), 4)), 2)


class F16(FeatureExtractor):
    """对于有限长度信号序列时域幅值，求余隙因子--最大值/平方和均值
    """

    @numba.jit(cache=True,parallel=True)
    def run(self, series):
        return round(np.max(series) / np.mean(np.square(series)), 2)


class F17(FeatureExtractor):
    """对于有限长度信号序列时域幅值，求脉冲因子--最大值/绝对均值
    """

    @numba.jit(cache=True,parallel=True)
    def run(self, series):
        return round(np.max(series) / np.mean(np.abs(series)), 2)


class F18(FeatureExtractor):
    """对于有限长度信号序列时域幅值，求能量因子--最大值/绝对均值
    """

    def __cycle_shit(self, series, k):
        return np.append(series[-1:], series[:-1])

    def __delta_x(self, sr):
        return np.square(sr[0]) - sr[1] * sr[2]

    @numba.jit(cache=True, parallel=True)
    def run(self, series):
        data = np.vstack((series, self.__cycle_shit(series, 1), self.__cycle_shit(series, -1)))
        delta_x = np.apply_along_axis(self.__delta_x, 0, data)
        return round(stats.moment(delta_x, 4) / np.square(stats.moment(delta_x, 2)), 2)


#  平均频率，中心频率，频率均方根，频率标准差 ,首先对时域信号进行傅里叶变换
class Signal2FourierDomain(object):
    def __init__(self, cache, fs=100):
        self.cache = cache
        self.fs = fs
        self.run_times = 0

    def transform(self, series):
        self.run_times += 1
        keys = self.cache.keys()
        if self.run_times in keys:
            if self.run_times%100==0:
                """print("使用FFT缓存结果- %s - "%self.run_times)"""
            return self.cache.get(self.run_times)
        else:
            self.cache[self.run_times] = self.__real_transform(series)
            return self.cache.get(self.run_times)

    @numba.jit(cache=True,parallel=True)
    def __real_transform(self, series):
        # Number of sample points
        N = len(series)
        # sample spacing
        T = 1.0 / self.fs
        x = np.linspace(0.0, N * T, N)
        yf = fftpack.fft(series)
        xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
        result_data = (xf, 2.0 / N * np.abs(yf[0:N // 2]))

        return result_data


# 所有的具体特征提取器都是无状态的，不会进行状态维护 #
# 频域特征计算
class F19(FeatureExtractor):
    """对于有限长度信号序列时域幅值，求平均功率
    """
    def __init__(self):
        self.transformer = None
        self.run_times=0

    def set_pre_transformer(self, transformer):
        self.transformer = transformer
        return self

    @numba.jit(cache=True,parallel=True)
    def run(self, series):
        self.run_times += 1
        if self.run_times%100 == 0:
            """print("the F19 data %s: "%self.run_times,series[0:5])"""
        result_data = self.transformer.transform(series)
        return round(np.mean(result_data[1]), 2)


class F20(FeatureExtractor):
    def __init__(self):
        self.transformer = None

    def set_pre_transformer(self, transformer):
        self.transformer = transformer
        return self

    @numba.jit(cache=True,parallel=True)
    def run(self, series):
        result_data = self.transformer.transform(series)
        return round(np.dot(result_data[0], result_data[1]) / np.sum(result_data[1]), 2)


class F21(FeatureExtractor):
    def __init__(self):
        self.transformer = None

    def set_pre_transformer(self, transformer):
        self.transformer = transformer
        return self

    @numba.jit(cache=True,parallel=True)
    def run(self, series):
        result_data = self.transformer.transform(series)
        return round(np.sqrt(np.dot(np.square(result_data[0]), result_data[1]) / np.sum(result_data[1])), 2)


class F22(FeatureExtractor):
    def __init__(self):
        self.transformer = None

    def set_pre_transformer(self, transformer):
        self.transformer = transformer
        return self

    @numba.jit(cache=True,parallel=True)
    def run(self, series):
        result_data = self.transformer.transform(series)
        # print(result_data)
        fc = np.dot(result_data[0], result_data[1]) / np.sum(result_data[1])
        # print(result_data[0].__class__)
        # print(np.square(result_data[0]-fc,2))
        return round(np.sqrt(np.dot(np.square(result_data[0] - fc), result_data[1]) / np.sum(result_data[1])), 2)


class F23(FeatureExtractor):
    """对于有限长度信号序列时域幅值，求脉冲因子--最大值/绝对均值
    """
    @numba.jit(cache=True)
    def run(self, series):
        return round(series[-1]-series[-4], 2)


class F24(FeatureExtractor):
    """对于有限长度信号序列时域幅值，求脉冲因子--最大值/绝对均值
    """
    @numba.jit(cache=True)
    def run(self, series):
        return round(series[-1]-series[-6], 2)


class F25(FeatureExtractor):
    """对于有限长度信号序列时域幅值，求脉冲因子--最大值/绝对均值
    """
    @numba.jit(cache=True)
    def run(self, series):
        return round(series[-1]-series[-10], 2)


class F26(FeatureExtractor):
    """对于有限长度信号序列时域幅值，求脉冲因子--最大值/绝对均值
    """
    @numba.jit(cache=True)
    def run(self, series):
        return round(series[-1]-series[-20], 2)


class F27(FeatureExtractor):
    """对于有限长度信号序列时域幅值，求脉冲因子--最大值/绝对均值
    """
    @numba.jit(cache=True)
    def run(self, series):
        return round(series[-1]-series[-40], 2)


class F28(FeatureExtractor):
    """对于有限长度信号序列时域幅值，求脉冲因子--最大值/绝对均值
    """
    @numba.jit(cache=True)
    def run(self, series):
        return round(series[-1]-series[-100], 2)


#################################################################
class ExtractorManager:
    def __init__(self):
        from boltons.cacheutils import LRU
        self.extractors = dict()
        self.cache = LRU(max_size=50000)

    def make_extractors(self):
        self.extractors["F01"] = F1()
        self.extractors["F02"] = F2()
        self.extractors["F03"] = F3()
        self.extractors["F04"] = F4()
        self.extractors["F05"] = F5()
        self.extractors["F06"] = F6()
        self.extractors["F07"] = F7()
        self.extractors["F08"] = F8()
        self.extractors["F09"] = F9()
        self.extractors["F10"] = F10()
        self.extractors["F11"] = F11()
        self.extractors["F12"] = F12()
        self.extractors["F13"] = F13()
        self.extractors["F14"] = F14()
        self.extractors["F15"] = F15()
        self.extractors["F16"] = F16()
        self.extractors["F17"] = F17()
        self.extractors["F18"] = F18()
        self.extractors["F19"] = F19().set_pre_transformer(Signal2FourierDomain(self.cache))
        self.extractors["F20"] = F20().set_pre_transformer(Signal2FourierDomain(self.cache))
        self.extractors["F21"] = F21().set_pre_transformer(Signal2FourierDomain(self.cache))
        self.extractors["F22"] = F22().set_pre_transformer(Signal2FourierDomain(self.cache))
        self.extractors["F23"] = F23()
        self.extractors["F24"] = F24()
        self.extractors["F25"] = F25()
        self.extractors["F26"] = F26()
        self.extractors["F27"] = F27()
        self.extractors["F28"] = F28()
        self.extractors['Identity'] = Identity()
        return self

    def get_extractor(self, key):
        return self.extractors.get(key)

    def add_extractor(self, key, extractor):
        self.extractors.update({str(key): extractor})


from pandas import DataFrame
from ..commons.common import JustTransformer,NamedDataFrame
from ..commons.utils import NamedDataFrame2CSV


# TODO(yu):这里又两种交叉模式，还有一种完全展开形式; a.一个特征处理所有列，多个特征重复然后合并；
# b.一列同时计算除多个特征，然后多列合并；c.特征和列完全交叉展开笛卡儿积，每个对应一个列转换元组；
# a符合基本设计可以利用ColumnTransformer，导致会多次; b特征计算可以共享FFT计算结果FFT,耦合度高,实现麻烦些; c最不划算的做法FFT
# 为了方便，选a，显然c最灵活，可以任意指定列和特征，而a.b只能灵活指定一维。
# FeatureUnion主要解决的是多种特征的合并，ColumnTransformer主要解决列的指定问题，而Pipeline主要解决竖直方向连接的问题
# 三者结合很有用,但是ColumnTransformer 似乎可以实现 FeatureUnion 的功能？ColumnTransformer似乎可以是实现都中情况，
# 通过列与特征交叉构建多个元组元素的列表。一下子算全部特征，或者一下子算全部列，都只剩下一维的自由度，因此本处采用，但特征与
# ColumnTransformer结合指定列，获得两个维度的自由度。

# 采用闭包持有特征提取器之后返回一个函数，并且可接受输入的数据
# 还必须将参数封装在转换器内不能用闭包进而用FunctionTransformer，因为交叉搜索的时候从这里取出
# 还是采用ColumnTransformer的数组结构，多各特征变换连接方便，而在采用自定义的NamedDataFrame可直接使用的包组建会减少
class SignalFeatureExtractor(JustTransformer):
    def __init__(self, extractor, start=0, window_size=512, step=60):
        self.extractor = extractor
        self.start = start
        self.window_size = window_size
        self.step = step

    @staticmethod
    def _cal_array_like(extractor, array, start, step, window_size):
        """:arg array: n维数组"""
        steps = [(it, it+window_size) for it in range(start, array.shape[0]-window_size, step)]

        def extract_one_column(_extractor, _one_d_array, _steps):
            return list(map(lambda idx: _extractor.run(_one_d_array[idx[0]:idx[1]]), _steps))
        result = [extract_one_column(extractor, array.values[:, it], steps) for it in range(array.shape[1])]
        result = np.array(result).T
        return result

    #  @NamedDataFrame2CSV(save_or_not=True)
    def transform(self, df: DataFrame):
        result = self._cal_array_like(self.extractor, df, self.start, self.step, self.window_size)
        result_df = DataFrame(data=result, columns=df.columns)
        return result_df



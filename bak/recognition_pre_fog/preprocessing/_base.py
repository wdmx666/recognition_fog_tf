"""
@Name:        _base
@Description: ''
@Author:      Lucas Yu
@Created:     2018/11/15
@Copyright:   (c) GYENNO Science,Shenzhen,Guangdong 2018
@Licence:
"""
import numpy
import functools
from sklearn import preprocessing
from pandas import DataFrame, Interval
from typing import Dict, List,Union,Tuple


# 尽量不去实现transformer,因为个人实现的transformer性能往往不全
# 原始数据过滤
# FunctionTransformer 不带参数，会有较大不方便。
# 处理程序，尽量不要带过于复杂的数据结构的信息，否则算法的通用会降低
def filter_data(x: numpy.ndarray)->numpy.ndarray:      # 传入的全部数组
    def between_range(array: numpy.ndarray)-> List[float]:
        itv = Interval(round(numpy.quantile(array, 0.005), 2), round(numpy.quantile(array, 0.995), 2), closed='both')
        return list(map(lambda num: num if num in itv else (itv.left if num < itv.left else itv.right), array))
    result = [between_range(x[:, it]) for it in range(x.shape[1])]
    result = numpy.array(result).T
    return result


def dynamic_scale_data(x: numpy.ndarray)->numpy.ndarray:
    def scale(array: numpy.ndarray)->numpy.ndarray:
        if abs(array.max() / array.min()) > 5000:
            scale_data = preprocessing.quantile_transform(array.reshape(-1, 1))   # tmd 非要2D数组
            print("dynamic_quantile_transform", array.shape)
            return scale_data
        else:
            scale_data = preprocessing.scale(array.reshape(-1, 1))
        return scale_data.ravel()
    result = numpy.column_stack([scale(x[:, it].ravel()) for it in range(x.shape[1])])
    return result


def add_weight(arr: numpy.ndarray) -> numpy.ndarray:
    arr.fill(1)
    return functools.reduce(lambda x, y: x*y, arr.T)



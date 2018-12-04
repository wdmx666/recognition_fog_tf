"""
@Name:        _base
@Description: ''
@Author:      Lucas Yu
@Created:     2018/11/15
@Copyright:   (c) GYENNO Science,Shenzhen,Guangdong 2018
@Licence:
"""
import pandas
from schema import Schema,Use,And,Or
from typing import Tuple
import sklearn


def _safe_split(data, feature_cols, target_col, weight_col, indices=None)->Tuple:
    data = Schema(pandas.DataFrame).validate(data)
    vfn = lambda y: len(set(y)-set(data.columns)) == 0
    feature_cols = And([str], vfn, error="列名不正确,请输入正确列名").validate(feature_cols)
    weight_col = And('weight', lambda x: x in data.columns, error="列名不正确,请输入正确列名").validate(weight_col)
    target_col = And(lambda x: x in data.columns, error="列名不正确,请输入正确列名").validate(target_col)
    indices = indices if indices else list(range(data.shape[0]))
    sklearn.utils.check_consistent_length(data, indices)
    #print(f"indice---> {indices,data.columns}")
    df = data.iloc[indices, :]
    return df.loc[:, feature_cols], df.loc[:, target_col], df.loc[:, weight_col]

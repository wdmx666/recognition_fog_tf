"""
@Name:        comm
@Description: ''
@Author:      Lucas Yu
@Created:     2018/11/6
@Copyright:   (c) GYENNO Science,Shenzhen,Guangdong 2018
@Licence:
"""

from sklearn.base import BaseEstimator,TransformerMixin
import path
import warnings


from typing import TypeVar, Generic,NamedTuple,Optional,Any,List,overload
from logging import Logger
from pandas import DataFrame

T = TypeVar('T')

class LoggedVar(Generic[T]):
    def __init__(self, value: T, name: str, logger: Logger) -> None:
        self.name = name
        self.logger = logger
        self.value = value

    def set(self, new: T) -> None:
        self.log('Set ' + repr(self.value))
        self.value = new

    def get(self) -> T:
        self.log('Get ' + repr(self.value))
        return self.value

    def log(self, message: str) -> None:
        self.logger.info('%s: %s', self.name, message)


class JustTransformer(BaseEstimator, TransformerMixin):

    def check_parameters(self):
        for it in vars(self):
            if getattr(self, it) is None:
                raise ValueError("%s属性为空" % it)

    def get_output_destination(self, filename=None):
        p = path.Path('../data/%s' % self.__class__.__name__).abspath()
        if not filename:
            return p
        else:
            if not p.exists():
                p.makedirs_p()
            return p.joinpath(filename)

    def fit(self, x, y=None, **fit_params):
        print(self.__class__.__name__, "fit什么都不做，直接返回self!")
        return self

    def transform(self, x: Optional[T])->Any:
        pass


NamedDataFrame = NamedTuple('NamedDataFrame', [('name', str), ('dataframe', DataFrame)])

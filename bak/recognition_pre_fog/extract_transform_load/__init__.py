"""
@Name:        __init__.py
@Description: ETL，是英文 Extract-Transform-Load 的缩写，用来描述将数据从来源端经过抽取（extract）、
              交互转换（transform）、加载（load）至目的端的过程。这个过程不会涉及到 ML中的数据泄露的问题。
@Author:      Lucas Yu
@Created:     2018-11-10
@Copyright:   ©GYENNO Technology,Shenzhen,Guangdong 2018
@Licence:
"""

from .preparation import ExtractorMetaData, MetaDataLoader, SignalMarkDataLoader,ConfigLoader,SignalMarkNamedDataFrameLoader
from ._extractor import SignalDataExtractor, VideoMarkDataExtractor
from ._transform import DropDuplicateFillLostTimePoint, SignalMarker


"""为了避免过于复杂数据结构的检查，一方面，通过自定义类型来缓解类型不安全问题，另一方面将数据放在配置文件中来避免问题
   而不是由程序员从方法中通过参数传入。
"""
"""因为缓存的原因，joblib导入会出问题，特别是模块位置名称发生变化时要清除缓存
"""
__all__ = ['ExtractorMetaData', 'MetaDataLoader', 'SignalDataExtractor', 'VideoMarkDataExtractor',
           'DropDuplicateFillLostTimePoint', 'SignalMarker', 'SignalMarkDataLoader','ConfigLoader',
           'SignalMarkNamedDataFrameLoader']

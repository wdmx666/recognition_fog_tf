"""
@Name:        app_cmd
@Description: ''
@Author:      Lucas Yu
@Created:     2018-11-11
@Copyright:   ©GYENNO Technology,Shenzhen,Guangdong 2018
@Licence:
"""
import path
from pyhocon import ConfigFactory
from sklearn.pipeline import make_pipeline
from sklearn.utils import Parallel, delayed, Memory
from recognition_pre_fog.extract_transform_load import MetaDataLoader, SignalDataExtractor,VideoMarkDataExtractor
from recognition_pre_fog.extract_transform_load import DropDuplicateFillLostTimePoint, SignalMarker

memory = Memory(location='../data/cache')


def transform_one(transformer, X):
    return transformer.transform(X)


cache_fit_transform_one = memory.cache(transform_one)


if __name__ == "__main__":
    parallel = Parallel(n_jobs=6)
    meta_data = MetaDataLoader.load()
    sde = SignalDataExtractor()
    ddfltp = DropDuplicateFillLostTimePoint()
    vde = VideoMarkDataExtractor()
    sm = SignalMarker()
    pipe_s = make_pipeline(sde, ddfltp, sm)
    #pipe_s = make_pipeline(sm)

    # res1 = parallel(delayed(cache_fit_transform_one)(sde, it) for it in meta_data.signal_files_info)
    # vde = VideoMarkDataExtractor()
    res2 = parallel(delayed(cache_fit_transform_one)(vde, it) for it in meta_data.video_mark_files_info['workbooks'])
    #res3 = parallel(delayed(cache_fit_transform_one)(pipe_s, it) for it in meta_data.signal_files_info)

    for it in meta_data.signal_files_info:
        print(pipe_s.transform(it))
    # #print(res2)
    #
    # for k in vars(meta_data): print(k, '->', getattr(meta_data,k))
    # print("==================================")
    # for i,j in meta_data.video_mark_files_info.items():
    #     print(i,'-> ',j)
    # print("==========-----------------------===========")
    # for i,j in meta_data.video_mark_sheet_info.items():
    #     print(i,'-> ',j)
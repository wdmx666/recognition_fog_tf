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
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Parallel, delayed, Memory
from recognition_pre_fog.extract_transform_load import SignalMarkDataLoader,ConfigLoader
from recognition_pre_fog.preprocessing import SignalStatusReMark
from recognition_pre_fog.commons.common import NamedDataFrame



# TODO(yu): pipeline这个实现设置参数好丑陋，每个对象有唯一名字为什么要那么深的__引用，维护一个对象空间OK?
memory = Memory(location='../data/cache')


def transform_one(transformer, X):
    return transformer.transform(X)


cache_fit_transform_one = memory.cache(transform_one)


if __name__ == "__main__":
   #parallel = Parallel(n_jobs=6)
   cfg = ConfigLoader.load(meta_data_path='doc/para.conf')
   # data = SignalMarkDataLoader.load(r'../data/SignalMarker')
   # data =NamedDataFrame('all_data', data)
   # srs = SignalStatusReMark()
   colt = ColumnTransformer([('standardscaler', StandardScaler(), cfg['SIGNAL_COLS_NAME']['data'])],remainder='passthrough')
   #print(srs.get_params())
   #srs.set_params(status_definition={"fog":["A","B","C","D","F"],"normal":["H","I","J","K"],"undefined":["N"]})
   #pipe = make_pipeline(srs)
   #colt = make_column_transformer((slice(0,54,1),srs))
   #result_1 = srs.transform(data)

   print(colt.get_params())
   # print(data.shape)
   # print(data.columns)
   # #print(pipe.get_params())
   #print(pipe.get_params(deep=False))
  # pipe.set_params(signalstatusremark__pre_fog_time_len=10000)
   #print(pipe.get_params(deep=False))
   #print(make_pipeline(colt).get_params())
   # for it in data:
   #     print(it.name, it.dataframe.shape)
       #srs.transform(it).dataframe.head(3)


    #pipe_s = make_pipeline(sm)

    # res1 = parallel(delayed(cache_fit_transform_one)(sde, it) for it in meta_data.signal_files_info)
    # vde = VideoMarkDataExtractor()
    #res2 = parallel(delayed(cache_fit_transform_one)(vde, it) for it in meta_data.video_mark_files_info['workbooks'])
    #res3 = parallel(delayed(cache_fit_transform_one)(pipe_s, it) for it in meta_data.signal_files_info)


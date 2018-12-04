
import path
import joblib
from ..commons.common import MyProcessor
from ..client.proj_config import MyNode
from ..s1_etl_signal_video.preparation import SignalETLConfig
from ..s3_signal_remark_extract.preparation import SignalRemarkExtractConfig
from ..s4_re_etl_feature.preparation import FeatureReETLConfig
from ..s5_train_validation.preparation import ValidationConfig
from ..commons.scheduler import AppGraph,AppSession


# 尽量使用不可变对象，减少副作用
class ExtractParaGridSearchCV(MyProcessor):  # only封装逻辑
    def __init__(self, name=None, dependencies=None, reset=False):
        super().__init__(name, dependencies, reset)
        self.para.setdefault('param_grid', None)

    def process(self):
        p1 = SignalETLConfig.signalETLInit()
        p2 = SignalETLConfig.signalETL()
        p3 = SignalETLConfig.videoMarkETL()
        p4 = SignalETLConfig.signalMark()

        for para_it in self.para['param_grid']:
            p5 = SignalRemarkExtractConfig.remarkScaleFeature(name='RemarkScaleFeature'+'_'+para_it['doe_id'])
            p5_sv = p5.para['RemarkScaleFeatureService']
            p5_sv.set_para_with_prop({"pre_fog_time_len": para_it['preFoG duration']})
            wd_para = {"start": 0, "ksize": para_it['window size'], "step": para_it['step']}
            p5_sv.get_para('window').set_para(**wd_para)

            p6 = FeatureReETLConfig.scaleFeatureOneByOne(dependencies=p5.name)

            p7 = ValidationConfig.validationProcessor(dependencies=p6.name)
            p7.set_para_with_prop({'param_id': para_it['doe_id']})

            graph, sess = AppGraph().add_processors([p1,p2,p3,p4,p5,p6,p7]), AppSession()

            sess.run(graph)

        print("over")


class FeatureGridSearchCV(MyProcessor):  # only封装逻辑
    def __init__(self, name=None, dependencies=None, reset=False):
        super().__init__(name, dependencies, reset)
        self.para.setdefault('param_grid', [])
        self.para.setdefault('FeatureRanker', None)

    def prepare(self):
        df_fi = self.para["FeatureRanker"].calculate(None)
        for k in list(range(1, 5))+list(range(5, 600, 40)):
            cols = df_fi["feature_name"][0:k].tolist()
            self.para['param_grid'].append((k, cols))

    def process(self):  # 到这一步已经只需要指定的输入数据和参数进行计算了，其他逻辑它们不管
        p1 = SignalETLConfig.signalETLInit()
        p2 = SignalETLConfig.signalETL()
        p3 = SignalETLConfig.videoMarkETL()
        p4 = SignalETLConfig.signalMark()

        for k, cols in self.para['param_grid']:
            p5 = SignalRemarkExtractConfig.remarkScaleFeature(name='RemarkScaleFeature')
            p5_sv = p5.para['RemarkScaleFeatureService']
            p5_sv.set_para_with_prop({"pre_fog_time_len": 600})
            wd_para = {"start": 0, "ksize": 600, "step": 10}
            p5_sv.get_para('window').set_para(**wd_para)

            p6 = FeatureReETLConfig.scaleFeatureOneByOne(dependencies=p5.name)

            p7 = ValidationConfig.validationProcessor(dependencies=p6.name)
            p7.para['CrossValidator'].para["TrainTestSplitValidator"].para["data_maker"].para["input_cols"]["feature_cols"] = cols

            p7.set_para_with_prop({'param_id': k})

            graph, sess = AppGraph().add_processors([p1,p2,p3,p4,p5,p6,p7]), AppSession()

            sess.run(graph)

        print("over")


####################################################################################################################

class ModelParaGridSearchCV(MyProcessor):
    def __init__(self, name=None, dependencies=None, reset=False):
        super().__init__(name, dependencies, reset)
        self.para.setdefault('graph', None)
        self.para.setdefault('param_grid', None)

    # def prepare(self):
    #     self.set_output_destination(path.Path(self.para.get("save_path")).joinpath(self.request.get_ID()))

    def process(self):  # 到这一步已经只需要指定的输入数据和参数进行计算了，其他逻辑它们不管
        p1 = SignalETLConfig.signalETLInit()
        p2 = SignalETLConfig.signalETL()
        p3 = SignalETLConfig.videoMarkETL()
        p4 = SignalETLConfig.signalMark()

        for para_it in self.para['param_grid']:
            p5 = SignalRemarkExtractConfig.remarkScaleFeature(name='RemarkScaleFeature')
            p5_sv = p5.para['RemarkScaleFeatureService']
            p5_sv.set_para_with_prop({"pre_fog_time_len": 600})
            wd_para = {"start": 0, "ksize": 600, "step": 10}
            p5_sv.get_para('window').set_para(**wd_para)

            p6 = FeatureReETLConfig.scaleFeatureOneByOne(dependencies=p5.name)

            p7 = ValidationConfig.validationProcessor(dependencies=p6.name)
            p7.set_para_with_prop({'param_id': para_it.pop('doe_id')})
            p7.para['CrossValidator'].para["TrainTestSplitValidator"].para["model"].set_params(**para_it)

            graph, sess = AppGraph().add_processors([p1, p2, p3, p4, p5, p6, p7]), AppSession()

            sess.run(graph)

        print("over")


#####################################################################################################################

class EventParaGridSearchCV(MyProcessor):
    def __init__(self, name=None, dependencies=None, reset=False):
        super().__init__(name, dependencies, reset)
        self.para.setdefault('param_grid')

    def set_para_with_prop(self, my_props):
        self.para.update(my_props)

    def prepare(self):
        self.set_output_destination(path.Path(self.para.get("save_path")).joinpath(self.request.get_ID()))

    def process(self):
        result = self.para["Calculator"].calculate((self.request.get_ID(), self.request.get_payload()[self.dependencies[0]]))
        joblib.dump(result, self.get_output_destination())





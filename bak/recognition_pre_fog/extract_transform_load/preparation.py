"""
@Name:        preparation
@Description: ''
@Author:      Lucas Yu
@Created:     2018/11/12
@Copyright:   (c) GYENNO Science,Shenzhen,Guangdong 2018
@Licence:
"""
import collections
import os, sys, path
import numpy
import itertools
import joblib
from pyhocon import ConfigFactory
from typing import Any, List, Dict
from pandas import DataFrame, read_csv
import schema
from ..commons.common import JustTransformer,NamedDataFrame


MetaData = collections.namedtuple('MetaData','signal_files_info,signal_sheet_info,video_mark_files_info,video_mark_sheet_info')


# 关系紧密代码要放一起，因为单独放也不能重用，因为他要配合其它代码；放一起高内聚
def _create_part_full_field_map(part_field :Dict[str, List[str]])->Dict[str, List[str]]:
    time_field = ["time16", "time10", "hour", "minute", "second", "ms"]
    return {part: time_field + [part + "_" + field for field in part_field[part] if field not in time_field] for part in
            part_field}


def _parse_dir(raw_data_path: str, sample_groups: List[str]) -> DataFrame:
    # 初始化相关量
    pid, data_group, sample_times, signal_file, video_result_file = [], [], [], [], []
    sample_paths = [path.Path(raw_data_path).joinpath(sample_group) for sample_group in sample_groups]
    # 遍历路径获取数据全名称
    for group_path in sample_paths:
        try:
            signal_files = group_path.files("[a-z0-9A-Z_]*.xlsx")
            signal_files = list(filter(lambda x: all([it not in x for it in ['result_1', '.mat']]), signal_files))
            video_mark_files = group_path.files("result_1.xlsx")
            signal_file.extend(signal_files)
            video_result_file.extend(video_mark_files * len(signal_files))
        except FileNotFoundError as e:
            print(FileNotFoundError("没有找到文件，请核查后再启动！"),e)
    if not (signal_file or video_result_file):
        try:
            sys.exit(1)
        except SystemExit as e:
            print("没有待处理的文件")

    c: itertools.count = itertools.count(1)
    for temp_fp in signal_file:
        filepath, tempfilename = os.path.split(temp_fp)
        filename, extension = os.path.splitext(tempfilename)
        data_group.append(os.path.basename(filepath))
        sample_times.append(filename)
        pid.append('pd{:0>6d}'.format(next(c)))
    file_df = DataFrame(data=numpy.array([pid, signal_file, video_result_file, data_group, sample_times]).transpose(),
                               columns=['id','signal_file','video_result_file','data_group','sample_times'])
    return file_df


def _prepare_signal_meta_data(raw_data: List[Dict[str, str]], transform_matrix:Dict[str, str], sheet_part:Dict[str, List[str]])->List[Dict[str, str]]:
    """{'id': 'pd000001','part': [a,b,c], 'transform_matrix':'path', 'signal_file': 'path','video_result_file': 'path',
        'data_group': '20161223', 'sample_times': '1_1'} 记录数据的数据结构说明"""
    full_description = []
    for it in raw_data:
        description = dict()
        description['part'] = sheet_part[it['id']]
        description['transform_matrix'] = transform_matrix[it['id']]
        description.update(it)
        full_description.append(description)
    return full_description


# 类型不安全，因为数据不一定有如下字段
def _prepare_video_mark_meta_data(data: DataFrame)->Dict[str,Any]:
    video_mark_description = {}
    sample_groups, samples, sample_times, result_names = {}, {}, {}, {}

    def output(name_info=None, new_name=None)->str:
        """输入和输出有关系"""
        info = [name_info['id'], "==", name_info['data_group'], "_", name_info['sample_times'],"_result_1.csv"]
        return ''.join(info)

    for idx in data.index:
        sample_groups.update({data.loc[idx, 'video_result_file']: data.loc[idx, 'data_group']})
        result_names.update({output(name_info=data.loc[idx].to_dict()): data.loc[idx, 'video_result_file']})
        samples[data.loc[idx, 'video_result_file']] = set()
        sample_times[data.loc[idx, 'video_result_file']] = []
    for idx in data.index:
        samples[data.loc[idx, 'video_result_file']].add(data.loc[idx, 'sample_times'].split("_")[0])
        sample_times[data.loc[idx, 'video_result_file']].append(data.loc[idx, 'sample_times'])

    video_mark_description['workbooks'] = list(sample_groups.keys())
    video_mark_description['sample_groups'] = sample_groups
    video_mark_description['samples'] = samples
    video_mark_description['sample_times'] = sample_times
    video_mark_description['result_names'] = result_names
    video_mark_description['workbooks'] = set(result_names.values())
    return video_mark_description


# 程序是如此的与数据结构关系紧密
class ExtractorMetaData(JustTransformer):
    __slots__ = ["data_description_path"]
    """混乱的文件系统结构，即文件在文件系统中乱放；混乱的文件结构，即文件内部组织混乱；这都十分可恶!合理将配置信息分类分层。"""
    def __init__(self, data_description_path=None):
        if not data_description_path:
            self.data_description_path = path.Path(__file__).parent.parent.parent.joinpath('doc/data_description.conf')
            print(self.data_description_path)
        else:
            self.data_description_path: str = os.path.abspath(data_description_path)

    def transform(self, conf=None) -> MetaData:

        conf = ConfigFactory.parse_file(os.path.abspath(self.data_description_path))
        df0 = _parse_dir(conf['RAW_DATA_PATH'], conf['SAMPLE_GROUPS'])

        signal_files_info = _prepare_signal_meta_data(df0.to_dict('records'), conf["TRANSFORM_MATRIX"], conf["SHEET_PART"])
        signal_sheet_info = dict(part_fields=_create_part_full_field_map(conf['PART_FIELD']), no_signal_cols=conf['NO_SIGNAL_COLS'])

        video_mark_files_info = _prepare_video_mark_meta_data(df0)
        video_mark_sheet_info = dict(
            sheet_structure={"repeat_step": 5, "detail_data_start": (4, 2), "video_time_start": (1, 2),
                             "confidence_start": (3, 5), "simulation_effect_start": (37, range(3))},
            unit_map={"video_result": DataFrame(), "start_time": 0, "unit_confidence": 0, "video_len": 0},
            video_data_field=["label_start_time", "label_end_time", "gait_type", "label_confidence", "stimulate_type"])

        result_all = MetaData(signal_files_info, signal_sheet_info, video_mark_files_info, video_mark_sheet_info)
        joblib.dump(result_all, self.get_output_destination())
        return result_all


class MetaDataLoader(object):
    @staticmethod
    def load(meta_data_path: str=None)->MetaData:
        if not meta_data_path:
            meta_data_path = path.Path(__file__).parent.parent.parent.joinpath('data/ExtractorMetaData')
        else:
            meta_data_path: str = os.path.abspath(meta_data_path)

        if path.Path(meta_data_path).exists():
            print(path.Path(meta_data_path).abspath())
            data = joblib.load(path.Path(meta_data_path).abspath())
            print(meta_data_path)
            return data
        else:

            return ExtractorMetaData().transform()


class ConfigLoader(object):
    @staticmethod
    def load(meta_data_path: str = 'doc/para.conf'):
        parent_path = path.Path(__file__).parent.parent.parent
        meta_data_path = parent_path.joinpath(meta_data_path)
        try:
            cfg = schema.Schema(schema.Use(lambda it: ConfigFactory.parse_file(it))).validate(meta_data_path)
        except FileNotFoundError as e:
            print(f'validation error {e}')
            sys.exit(1)
        return cfg


class SignalMarkNamedDataFrameLoader:
    @staticmethod
    def load(data_path: str)->List[NamedDataFrame]:
        result_dfs: List[NamedDataFrame] = []
        file_names = schema.Use(lambda x: path.Path(x).files()).validate(data_path)
        for it in file_names:
            df = read_csv(open(it))
            result_dfs.append(NamedDataFrame(it.name, df))
        return result_dfs


# 为配合框架的使用模式，将数据按组连接起来
class SignalMarkDataLoader:
    @staticmethod
    def load(data_path: str)->DataFrame:
        result_dfs: DataFrame = DataFrame()
        file_names = schema.Use(lambda x: path.Path(x).files()).validate(data_path)
        for it in file_names:
            df = read_csv(open(it))
            df['sample_name'] = it.name
            result_dfs = result_dfs.append(df)
        return result_dfs

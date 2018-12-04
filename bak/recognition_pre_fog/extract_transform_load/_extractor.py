"""
@Name:        algo_core
@Description: '从乱七八糟的数据源处提取数据'
@Author:      Lucas Yu
@Created:     2018/11/5
@Copyright:   ©GYENNO Technology,Shenzhen,Guangdong 2018
@Licence:
"""

import numpy
import itertools
from typing import Any, List, Dict
from pandas import DataFrame, read_excel, read_csv
from .preparation import MetaDataLoader
from ..commons.common import JustTransformer, NamedDataFrame


__all__ = ['SignalDataExtractor', 'VideoMarkDataExtractor']


# 一般数据组织不太适合ColumnTransformer,ColumnTransformer没有完全初始化其输入
class SignalDataExtractor(JustTransformer):
    def __init__(self, meta_loader=MetaDataLoader)->None:
        """数据结构相关数据融合.
        Parameters
        ----------
        part_fields: {'A100': ['time16', 'time10', 'hour', ..., 'A100_accelerometer_x', 'A100_accelerometer_y',...],
                     'A200': ['time16', 'time10', 'hour', 'minute', 'second', 'ms', 'A200_accelerometer_x...}
            no_signal_cols: ["time16", "hour", "minute", "second", "ms"]
        """
        self.signal_sheet_info: Dict[str, Any] = meta_loader.load().signal_sheet_info
        print("the instance is :", id(self.signal_sheet_info))

    def _merge_sheets(self, workbook_url: str, parts: List[str]=None)->DataFrame:
        df_map = read_excel(workbook_url, sheet_name=parts, header=None)
        c: itertools.count = itertools.count(0)
        result_df = DataFrame()
        for part in df_map.keys():
            print("part", part)
            df = df_map[part]
            df.columns = self.signal_sheet_info['part_fields'][part]
            if next(c) != 0:
                result_df = result_df.merge(df, on="time10", how="inner")
            else:
                result_df = df
        return result_df

    @staticmethod
    def _transform_coordinate(merged_df: DataFrame, transform_matrix: DataFrame, parts: List[str])->DataFrame:
        """必须使用列名称信息，所以使用df"""
        all_cols = []
        result_df = DataFrame(index=merged_df.index)
        for part in parts:
            tm_df = transform_matrix[transform_matrix["part"] == part]
            tm_df.drop(columns=["part"], inplace=True)
            cols = [col for col in merged_df.columns if col.find(part+"_") > -1]

            all_cols.extend(cols)
            result_df = result_df.join(DataFrame(numpy.matmul(merged_df[cols].values, tm_df.values),columns=cols),
                                       how="inner")
        for col in merged_df.columns:
            if col.find("foot") > -1:
                result_df[col] = round(result_df[col].apply(lambda x: 4878*(x if x>10 else merged_df[col].mean())**(-0.837)),2)
        return result_df

    def transform(self, record: Dict[str, Any])-> NamedDataFrame:
        self.check_parameters()
        print(record)
        merged_df_all = self._merge_sheets(record['signal_file'], record['part'])
        transform_matrix = read_csv(record['transform_matrix'])

        drop_cols = set(merged_df_all.columns).intersection(set(self.signal_sheet_info['no_signal_cols']))
        merged_df = merged_df_all.drop(columns=drop_cols, inplace=False)
        result_df = merged_df_all[['time10']]
        result_name = ''.join([record['id'], "==", record['data_group'], "_", record['sample_times'], ".csv"])
        result_df = result_df.join(self._transform_coordinate(merged_df, transform_matrix, record['part']), how="inner")
        result_df.to_csv(self.get_output_destination(result_name), index=False)
        return NamedDataFrame(result_name, result_df)


class VideoMarkDataExtractor(JustTransformer):

    def __init__(self, meta_loader=MetaDataLoader):
        # workbook内部sheet的信息
        self.video_mark_sheet_info: Dict[str, Any] = meta_loader.load().video_mark_sheet_info
        self.video_mark_files_info: Dict[str, Any] = meta_loader.load().video_mark_files_info

    def _parse_sheet(self, sheet, times):
        sheet_structure = self.video_mark_sheet_info['sheet_structure']
        start_col_no = (int(times) - 1) * sheet_structure["repeat_step"] + sheet_structure["detail_data_start"][1]
        end_col_no = start_col_no + sheet_structure["repeat_step"]
        start_row_no = sheet_structure["detail_data_start"][0]
        for i in range(500):
            try:
                if not str(sheet.iloc[i + start_row_no, start_col_no]).isdigit():
                    break
                else:
                    end_row_no = i + start_row_no
            except Exception as e:
                print(e)
        data = sheet.iloc[start_row_no:end_row_no + 1, start_col_no:end_col_no]
        data.columns = self.video_mark_sheet_info['video_data_field']
        # 将时间转化为十进制绝对值
        row_no = sheet_structure["video_time_start"][0]
        col_no = (int(times) - 1) * sheet_structure["repeat_step"] + sheet_structure["video_time_start"][1]

        data["start_time"] = (data["label_start_time"]) * 100 + 0 + int(str(sheet.iloc[row_no, col_no]), 16)
        data["end_time"] = (data["label_end_time"]) * 100 + 99 + int(str(sheet.iloc[row_no, col_no]), 16)
        print("+++++++++++++++++++++++++++++++++")

        # 获取本次采用整体置信度
        row_no = sheet_structure["confidence_start"][0]
        col_no = (int(times) - 1) * sheet_structure["repeat_step"] + sheet_structure["confidence_start"][1]
        data["unit_confidence"] = sheet.iloc[row_no, col_no]
        return data

    def transform(self, workbook: str)->List[NamedDataFrame]:
        sample_groups = self.video_mark_files_info['sample_groups']
        samples = self.video_mark_files_info['samples']
        sample_times = self.video_mark_files_info['sample_times']
        result_names = self.video_mark_files_info['result_names']
        result_book = read_excel(workbook, sheet_name=list(samples[workbook]), header=None)

        result = []
        for sample_time in sample_times[workbook]:
            sample_time_list = sample_time.split("_")
            result_df = self._parse_sheet(result_book[sample_time_list[0]], sample_time_list[1])
            con = filter(lambda it: it.find(sample_groups[workbook]) >= 0 and it.find(sample_time) >= 0, result_names)
            result_name = next(con)
            result.append(NamedDataFrame(result_name, result_df))
            result_df.to_csv(self.get_output_destination(result_name), index=False)
        return result


def _check_conf(conf):
    print(conf)


from functools import wraps


class NamedDataFrame2CSV(object):
    def __init__(self, save_or_not = False):
        self._save_or_not = save_or_not

    def __call__(self, func):
        @wraps(func)
        def wrapped_func(*args, **kwargs):
            obj = args[0]
            result = func(*args, **kwargs)
            if self._save_or_not:
                result.dataframe.to_csv(obj.get_output_destination(result.name), index=False)
                print(f"{result.name} --> 存储成功！")
            return result
        return wrapped_func


import inspect
def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]

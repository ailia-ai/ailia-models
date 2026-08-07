import numpy as np
from typing import Any

def numpy_type_to_builtin_type(obj: Any) -> Any:
    """numpy の型を Python の組み込み型に再帰的に変換する

    numpy のスカラー・配列を、対応する Python の組み込み型(int, float, bool, str, list, etc...) に変換する
    コンテナは中身を再帰的に処理する
    """
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return numpy_type_to_builtin_type(obj.tolist())
    if isinstance(obj, dict):
        return { numpy_type_to_builtin_type(k): numpy_type_to_builtin_type(v) for k, v in obj.items() }
    if isinstance(obj, list):
        return [numpy_type_to_builtin_type(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(numpy_type_to_builtin_type(v) for v in obj)
    if isinstance(obj, (set, frozenset)):
        return type(obj)(numpy_type_to_builtin_type(v) for v in obj)
    return obj


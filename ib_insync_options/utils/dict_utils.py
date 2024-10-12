import json
from enum import Enum
from typing import Dict


def _is_float(value: any) -> bool:
    return isinstance(value, float)


def round_dict_key_value(key: Enum, row: Dict, number_digits: int = 3) -> None:
    row[key.value] = round(row[key.value], number_digits)


def add_dict_key_value_to_agg_dict_if_float(
    key: Enum, row: Dict, total_row: Dict
) -> None:
    value = row[key.value]
    if isinstance(value, float):
        total_row[key.value] += value


def gen_json(obj):
    """
    Gen json from object, or nested-objects
    """
    return json.loads(json.dumps(obj, default=lambda o: getattr(o, "__dict__", str(o))))

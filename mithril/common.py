# Copyright 2022 Synnada, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass, field
from enum import IntEnum
from types import UnionType
from typing import Any, TypeVar


@dataclass
class CGenConfig:
    # Import configs
    HEADER_NAME: str = ""

    # Array configs
    ARRAY_NAME: str = ""

    # Function call configs
    IMPLICIT_BROADCAST_OPS: set[str] = field(default_factory=set)
    USE_OUTPUT_AS_INPUT: bool = False
    RETURN_OUTPUT: bool = False

    # Memory Management
    ALLOCATE_INTERNALS: bool = False


@dataclass
class PythonGenConfig:
    # Import configs
    SPECIFY_DEVICE: bool = False

    # Function call configs
    IMPLICIT_BROADCAST_OPS: set[str] = field(default_factory=set)


class PaddingType(IntEnum):
    VALID = 0
    SAME = 1


K = TypeVar("K")
V = TypeVar("V")


class BiMap(MutableMapping[K, V]):
    # Implements a bi-directional map for storing unique keys/values using two
    # dictionaries.
    # TODO: override __reversed__ for BiMap
    inverse: dict[V, K]
    _table: dict[K, V]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.inverse = inverse = {}
        self._table = table = dict(*args, **kwargs)
        for key, value in table.items():
            if value in inverse:
                raise ValueError(f"Value {value} maps to multiple keys")
            inverse[value] = key

    def __getitem__(self, key: K) -> V:
        return self._table[key]

    def __setitem__(self, key: K, value: V) -> None:
        if value in (inverse := self.inverse):
            existing_key = inverse[value]
            if key != existing_key:
                msg = f"Value {value} already exists with key {existing_key}"
                raise ValueError(msg)
        else:
            if key in (table := self._table):
                del inverse[table[key]]
            inverse[value] = key
            table[key] = value

    def __delitem__(self, key: K) -> None:
        del self.inverse[(table := self._table)[key]]
        del table[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self._table)

    def __len__(self) -> int:
        return len(self._table)


# Other utils
def find_dominant_type(lst: Any) -> type[int] | type[float] | type[bool]:
    # return dominant type of parameters in the list.
    # dominant type is referenced from numpy and in folloing order: bool -> int -> float
    # if any of the parameters are different from these three types, returns ValueError
    # if raise_error set to True. Otherwise returns this type.

    # Examples:
    # list contains both floats and ints -> return float
    # list contains both ints and bools -> return int
    # list contains only bools -> return bool
    # list contains all three of types -> return float

    if isinstance(lst, list | tuple):
        curr_val: type[bool] | type[int] | type[float] = bool
        if not lst:
            # Interpret empty list as float.
            # TODO: Backend array methods returns float arrays when
            # provided empty list as input argument. This is main reason
            # for this decision. Check if this is correct.
            return float
        for elem in lst:
            val = find_dominant_type(elem)
            if val is float:
                curr_val = float
            elif val is int:
                if curr_val is bool:
                    curr_val = int
            elif val is not bool:
                curr_val = val
                break
        return curr_val
    elif isinstance(lst, bool | float | int):
        return type(lst)

    raise ValueError(
        f"given input contains {type(lst)} type. Allowed types are: list, tuple, "
        "float, int, bool"
    )


def get_specific_types_from_value[T](value: Any, typ: type[T]) -> list[T]:
    """
    Recursively extracts all instances of a specified type from a nested structure.

    This function traverses through a nested structure (which can be a list,
    tuple, or dictionary) and collects all instances of the specified type.

    Args:
        value (Any): The input value which can be of any type, including nested lists,
        tuples, or dictionaries.
        typ (type[Typ]): The type to be extracted from the nested structure.

    Returns:
        list[Typ]: A list of all instances of the specified type found within the
        nested structure.
    """
    items = []
    if isinstance(value, typ):
        items.append(value)
    elif isinstance(value, list | tuple):
        for item in value:
            items += get_specific_types_from_value(item, typ)
    elif isinstance(value, dict):
        for val in value.values():
            items += get_specific_types_from_value(val, typ)
    return items


def contains_given_type(value: Any, typ: type | UnionType) -> bool:
    """
    Check if the given value or any nested value within it is of the specified type.

    This function traverses through the given value, which can be a single value,
    a list, a tuple, or a dictionary, and checks if any element within it matches
    the specified type.

    Args:
        value (Any): The value to be checked. It can be a single value, a list,
                     a tuple, or a dictionary.
        typ (type | UnionType): The type to check against.

    Returns:
        bool: True if any element within the value is of the specified type,
              False otherwise.
    """
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, typ):
            # If any item is of the specified type, return True
            return True
        if isinstance(current, list | tuple):
            stack.extend(current)
        elif isinstance(current, dict):
            stack.extend(current.values())
    return False

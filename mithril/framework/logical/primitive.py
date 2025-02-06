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

from collections.abc import KeysView

from ... import core
from ..common import (
    BaseKey,
)
from .model import ConnectionType, IOKey, Model
from .operator import Operator

__all__ = ["PrimitiveModel", "OperatorModel"]

ConstantType = float | int | core.Constant


class PrimitiveModel(Model):
    def __init__(
        self,
        formula_key: str,
        *,
        name: str | None = None,
        **kwargs: BaseKey,
    ) -> None:
        model = Operator(formula_key, self.class_name, **kwargs)
        super().__init__(name=name, enforce_jit=model._jittable)
        self._extend(model, self.wrap_keys(kwargs.keys()))

    @staticmethod
    def wrap_keys(kwargs: KeysView[str]) -> dict[str, ConnectionType]:
        return {key: IOKey(key, expose=True) for key in kwargs}

    @property
    def submodel(self) -> Operator:
        m = next(iter(self.dag.keys()))
        assert isinstance(m, Operator)
        return m


class OperatorModel(PrimitiveModel):
    def __init__(
        self,
        model: Operator,
        *,
        name: str | None = None,
    ) -> None:
        Model.__init__(self, name=name, enforce_jit=model._jittable)
        self._extend(model, self.wrap_keys(model.external_keys))

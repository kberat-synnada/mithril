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

import abc
import re
from collections.abc import Sequence
from copy import deepcopy
from types import UnionType
from typing import Any

from ..framework.common import (
    TBD,
    AllValueType,
    ConnectionData,
    GenericTensorType,
    IOHyperEdge,
    IOKey,
    Tensor,
    ToBeDetermined,
)
from ..framework.constraints import constrain_fn_dict
from ..framework.logical import essential_primitives
from ..models import (
    BaseModel,
    Connection,
    CustomPrimitiveModel,
    Model,
    models,
    primitives,
)
from ..models.train_model import TrainModel
from ..utils import model_conversion_lut
from ..utils.utils import PaddingType, convert_to_tuple

model_dict = {
    item[0].lower(): item[1]
    for item in models.__dict__.items()
    if isinstance(item[1], abc.ABCMeta) and issubclass(item[1], BaseModel)
}
model_dict |= {
    item[0].lower(): item[1]
    for item in primitives.__dict__.items()
    if isinstance(item[1], abc.ABCMeta) and issubclass(item[1], BaseModel)
}
model_dict |= {
    item[0].lower(): item[1]
    for item in essential_primitives.__dict__.items()
    if isinstance(item[1], abc.ABCMeta) and issubclass(item[1], BaseModel)
}

model_dict |= {"trainmodel": TrainModel}

__all__ = [
    "dict_to_model",
    "handle_dict_to_model_args",
    "dict_to_regularizations",
]
enum_dict = {"PaddingType": PaddingType}


def create_iokey_kwargs(info: dict[str, Any]) -> dict[str, Any]:
    kwargs = {}
    for arg in ["name", "value", "shape", "expose"]:
        if arg != "value" or info.get(arg) is not None:
            kwargs[arg] = info.get(arg)
    return kwargs


def dict_to_model(modelparams: dict[str, Any]) -> BaseModel:
    """Convert given dictionary to a model object.

    Parameter
    ----------
    modelparams : Dict[str, Any]
        A dict containing model name and arguments.

    Returns
    -------
    Model
        Instantiated model object with given arguments.
    """

    # TODO: Simplify dict_to_model and frameworks (remove tracking
    #  the extend order of models).
    if isinstance(modelparams, str):
        params = {"name": modelparams}
    elif "is_train_model" in modelparams:
        return dict_to_trainmodel(modelparams)
    else:
        params = deepcopy(modelparams)

    args: dict[str, Any] = {}
    if (connections := params.pop("connections", {})).keys() != (
        submodels := params.pop("submodels", {})
    ).keys():
        raise KeyError("Requires submodel keys and connections keys to be compatible!")

    if (model_name := params.get("name", None)) is None:
        raise Exception("No model type is specified!")
    elif model_name.lower() in model_dict:
        model_class = model_dict[model_name.lower()]
        args |= handle_dict_to_model_args(model_name, params.pop("args", {}))
        tuples = params.get("tuples", [])
        enums = params.get("enums", {})

        for k, v in args.items():
            if k in tuples:
                args[k] = convert_to_tuple(v)
            elif (enum_key := enums.get(k)) is not None:
                args[k] = enum_dict[enum_key][v]

        model = model_class(**args)

    else:  # Custom model
        args |= handle_dict_to_model_args(model_name, params.pop("args", {}))
        attrs = {"__init__": lambda self: super(self.__class__, self).__init__(**args)}
        model = type(model_name, (CustomPrimitiveModel,), attrs)()

    unnamed_keys = params.get("unnamed_keys", [])
    differentiability_info: dict[str, bool] = params.get("differentiability_info", {})
    assigned_shapes = params.get("assigned_shapes", {})
    assigned_constraints = params.get("assigned_constraints", {})
    canonical_keys = params.get("canonical_keys", {})

    submodels_dict = {}
    for m_key, v in submodels.items():
        m = dict_to_model(v)
        submodels_dict[m_key] = m
        mappings: dict[str, IOKey | float | int | list | tuple | str] = {}
        for k, conn in connections[m_key].items():
            if conn in unnamed_keys and k in m.input_keys:
                continue

            if isinstance(conn, str | float | int | tuple | list):
                mappings[k] = conn

            elif isinstance(conn, dict):
                if "connect" in conn:
                    key_kwargs = {}
                    if (key := conn.get("key")) is not None:
                        key_kwargs = create_iokey_kwargs(conn["key"])
                        key = IOKey(**key_kwargs)
                    mappings[k] = IOKey(
                        **key_kwargs,
                        connections={
                            getattr(submodels_dict[value[0]], value[1])
                            if isinstance(value, Sequence)
                            else value
                            for value in conn["connect"]
                        },
                    )
                elif "name" in conn:
                    key_kwargs = create_iokey_kwargs(conn)
                    mappings[k] = IOKey(**key_kwargs)

        assert isinstance(model, Model)
        model += m(**mappings)

    if "model" in canonical_keys:
        candidate_canonical_in = model.conns.get_connection(canonical_keys["model"][0])
        candidate_canonical_out = model.conns.get_connection(canonical_keys["model"][1])

        if candidate_canonical_in is not None:
            model._canonical_input = candidate_canonical_in
        if candidate_canonical_out is not None:
            model._canonical_output = candidate_canonical_out

    for key, value in differentiability_info.items():
        con = model.conns.get_connection(key)
        assert con is not None
        con.set_differentiable(value)

    if len(assigned_constraints) > 0:
        constrain_fn = assigned_constraints["fn"]
        if constrain_fn not in constrain_fn_dict:
            raise RuntimeError(
                "In the process of creating a model from a dictionary, an unknown"
                " constraint function was encountered!"
            )
        constrain_fn = constrain_fn_dict[constrain_fn]
        model.set_constraint(constrain_fn, keys=assigned_constraints["keys"])

    if len(assigned_shapes) > 0:
        model.set_shapes(dict_to_shape(assigned_shapes))

    return model


def model_to_dict(model: BaseModel) -> dict:
    if isinstance(model, TrainModel):
        return train_model_to_dict(model)

    model_name = model.__class__.__name__
    model_dict: dict[str, Any] = {"name": model_name}
    args = handle_model_to_dict_args(model_name, model.factory_args)
    if len(args) > 0:
        model_dict["args"] = args

    model_dict["assigned_shapes"] = {}
    model_dict["differentiability_info"] = {}  # TODO: save only assigned info not all!
    model_dict["assigned_constraints"] = {}

    for key, con in model.conns.all.items():
        data = con.metadata.data
        if isinstance(data, Tensor) and not con.is_key_autogenerated:
            model_dict["differentiability_info"][key] = data.differentiable

    for shape in model.assigned_shapes:
        model_dict["assigned_shapes"] |= shape_to_dict(shape)

    for constrain in model.assigned_constraints:
        model_dict["assigned_constraints"] |= constrain

    if (
        model_name != "Model"
        and model_name in dir(models)
        or model_name not in dir(models)
    ):
        return model_dict

    connection_dict: dict[str, dict] = {}
    canonical_keys: dict[str, tuple[str, str]] = {}
    submodels: dict[str, dict] = {}

    # IOHyperEdge -> [model_id, connection_name]
    submodel_connections: dict[IOHyperEdge, list[str]] = {}
    assert isinstance(model, Model)

    for idx, submodel in enumerate(model.dag.keys()):
        model_id = f"m_{idx}"
        submodels[model_id] = model_to_dict(submodel)

        # Store submodel connections
        for key in submodel._all_keys:
            submodel_connections.setdefault(
                submodel.conns.get_metadata(key), [model_id, key]
            )
        assert isinstance(model, Model)
        connection_dict[model_id] = connection_to_dict(
            model, submodel, submodel_connections, model_id
        )
        canonical_keys[model_id] = (
            submodel.canonical_input.key,
            submodel.canonical_output.key,
        )
    canonical_keys["model"] = (model.canonical_input.key, model._canonical_output.key)

    model_dict["submodels"] = submodels
    model_dict["connections"] = connection_dict
    model_dict["canonical_keys"] = canonical_keys
    return model_dict


def connection_to_dict(
    model: Model,
    submodel: BaseModel,
    submodel_connections: dict[IOHyperEdge, list[str]],
    model_id: str,
):
    connection_dict: dict[str, Any] = {}
    connections: dict[str, ConnectionData] = model.dag[submodel]

    for key, connection in connections.items():
        key_value: dict | None | str | AllValueType = None
        related_conn = submodel_connections.get(connection.metadata, [])
        is_valued = (
            connection.metadata.data.is_non_diff
            and connection.metadata.data.value != TBD
        )
        # Connection is defined and belong to another model
        if related_conn and model_id not in related_conn:
            key_value = {"connect": [related_conn]}
            if connection.key in model.output_keys:
                key_value["key"] = {"name": connection.key, "expose": True}
        elif is_valued and connection in model.conns.input_connections:
            val = connection.metadata.data.value
            assert not isinstance(val, ToBeDetermined)
            if connection.key.startswith("$"):
                key_value = val
            else:
                key_value = {"name": connection.key, "value": val, "expose": True}
        elif not connection.key.startswith("$"):
            if key in submodel.output_keys and connection.key in model.output_keys:
                key_value = {"name": connection.key, "expose": True}
            else:
                key_value = connection.key

        if key_value is not None:
            connection_dict[key] = key_value

    if submodel.canonical_input.key not in connection_dict:
        connection_dict[submodel.canonical_input.key] = ""

    return connection_dict


def train_model_to_dict(context: TrainModel) -> dict:
    context_dict: dict[str, Any] = {"is_train_model": True}
    context_dict["model"] = model_to_dict(context._model)

    losses = []
    regularizations = []
    for loss in context._losses:
        loss_dict: dict[str, Any] = {}
        loss_dict["model"] = model_to_dict(loss["loss_model"])
        loss_dict["reduce_steps"] = [
            model_to_dict(reduce_step) for reduce_step in loss["reduce_steps"]
        ]
        # TODO: check if get_local_key to get keys required?
        for key, value in loss["args"].items():
            if isinstance(value, Connection):
                # local_key = get_local_key(context._model, value)
                # loss["args"][key] = local_key
                loss["args"][key] = value.data.key

        if len(loss["args"]) > 0:
            loss_dict["args"] = loss["args"]
        losses.append(loss_dict)

    for regularization in context._regularizations:
        regularization_dict = {}
        regularization_dict["model"] = model_to_dict(regularization["reg_model"])
        regularization_dict["coef"] = regularization["coef"]
        regularization_dict["reg_key"] = regularization["reg_key"]
        for key, value in regularization["args"].items():
            if isinstance(value, Connection):
                # local_key = get_local_key(context._model, value)
                # regularization["args"][key] = local_key
                regularization["args"][key] = value.key
            elif isinstance(value, re.Pattern):
                regularization["args"][key] = {"pattern": value.pattern}

        if len(regularization["args"]) > 0:
            regularization_dict["args"] = regularization["args"]
        regularizations.append(regularization_dict)

    context_dict["losses"] = losses
    context_dict["regularizations"] = regularizations
    return context_dict


def dict_to_trainmodel(context_dict: dict) -> BaseModel:
    model = dict_to_model(context_dict["model"])
    assert isinstance(model, Model), "TrainModel requires a Model object!"

    context = TrainModel(model)
    for loss_dict in context_dict["losses"]:
        loss_model = dict_to_model(loss_dict["model"])
        reduce_steps = [
            dict_to_model(reduce_step) for reduce_step in loss_dict["reduce_steps"]
        ]
        loss_args = loss_dict["args"]
        context.add_loss(loss_model, reduce_steps, **loss_args)

    for regularization_dict in context_dict["regularizations"]:
        regularization_model = dict_to_model(regularization_dict["model"])
        coef = regularization_dict["coef"]
        reg_key = regularization_dict["reg_key"]
        regularization_args = {}
        for key, value in regularization_dict["args"].items():
            if isinstance(value, dict):
                regularization_args[key] = re.compile(value["pattern"])
            else:
                regularization_args[key] = value
        context.add_regularization(
            regularization_model, coef, reg_key, **regularization_args
        )

    return context


def handle_dict_to_model_args(
    model_name: str, source: dict[str, Any]
) -> dict[str, Any]:
    """This function converts model strings to model classes.

    Parameters
    ----------
    source : dict[str, Any]
        All arguments given as modelparams.
    """
    for key in model_conversion_lut.get(model_name.lower(), []):
        info = source[key]
        if isinstance(info, list):
            source[key] = [
                model_dict[k.lower()]() if isinstance(k, str) else dict_to_model(k)
                for k in info
            ]
        if isinstance(info, str):
            source[key] = model_dict[info.lower()]()
        if isinstance(info, dict):
            source[key] = dict_to_model(info)

    for key in source:
        if not isinstance(source[key], IOKey) and source[key] == "(Ellipsis,)":
            source[key] = ...

    for key, value in source.items():
        if isinstance(value, dict):
            shape_template: list[str | int | tuple] = []
            possible_types = None
            # Type is common for TensorType and Scalar.
            for item in value.get("type", []):
                # TODO: this is dangerous!!
                item_type: type = eval(item)
                if possible_types is None:
                    possible_types = item_type
                else:
                    possible_types |= item_type

            # TensorType.
            if "shape_template" in value:
                for item in source[key]["shape_template"]:
                    if "..." in item:
                        shape_template.append((item.split(",")[0], ...))
                    else:
                        shape_template.append(int(item))

                # assert possible_types is not None
                source[key] = IOKey(shape=shape_template, type=GenericTensorType)
                # TODO: Do not send GenericTensorType,
                # find a proper way to save and load tensor types.
            else:  # Scalar
                source[key] = IOKey(type=possible_types, value=source[key]["value"])
    return source


def handle_model_to_dict_args(
    model_name: str, source: dict[str, Any]
) -> dict[str, Any]:
    """This function converts model strings to model classes.

    Parameters
    ----------
    source : dict[str, Any]
        All arguments given as modelparams.
    """
    for key in model_conversion_lut.get(model_name.lower(), []):
        info = source[key]
        if isinstance(info, list):
            source[key] = [model_to_dict(k) for k in info]
        else:
            source[key] = model_to_dict(info)

    for key in source:
        if type(item := source[key]) is type(...):
            source[key] = "(Ellipsis,)"
        elif isinstance(item, IOKey):
            source[key] = item_to_json(source[key])
    return source


def dict_to_regularizations(
    regularizations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    reg_specs = []
    for reg in regularizations:
        if (regex := reg.get("regex")) is not None and any(
            isinstance(item, tuple) for item in reg["inputs"]
        ):
            raise Exception(
                "Regex style definitions are only valid for single input regularizers!"
            )

        inputs: list = []
        model = model_dict[reg["model"]](**reg.get("args", {}))

        for idx, item in enumerate(reg["inputs"]):
            if isinstance(item, list):
                inputs.append(tuple(item))
            elif regex is None:
                inputs.append(item)
            # If regex key provided, check its status for the
            # corresponding index. If True, convert string into
            # regex Pattern object.
            else:
                if regex[idx]:
                    inputs.append(re.compile(item))
                else:
                    inputs.append(item)

        reg_spec: dict[str, Any] = {}
        reg_spec["model"] = model
        reg_spec["inputs"] = inputs
        if (model_keys := reg.get("model_keys")) is not None:
            reg_spec["model_keys"] = (
                tuple(model_keys) if isinstance(model_keys, list) else (model_keys,)
            )

        reg_specs.append(reg_spec)
    return reg_specs


def shape_to_dict(shapes):
    shape_dict = {}
    for key, shape in shapes.items():
        shape_list = []
        for item in shape:
            if isinstance(item, tuple):  # variadic
                shape_list.append(f"{item[0]},...")
            else:
                shape_list.append(item)
        shape_dict[key] = shape_list
    return shape_dict


def dict_to_shape(shape_dict):
    shapes: dict[str, list[int | tuple]] = {}
    for key, shape_list in shape_dict.items():
        shapes[key] = []
        for shape in shape_list:
            if isinstance(shape, str) and "..." in shape:
                shapes[key].append((shape.split(",")[0], ...))
            else:
                shapes[key].append(shape)

    return shapes


def type_to_str(item):
    if "'" in str(item):
        return str(item).split("'")[1]
    return str(item)


def item_to_json(item: IOKey):
    # TODO: Currently type is not supported for Tensors.
    # Handle This whit conversion test updates.
    result: dict[str, Any] = {}
    if not isinstance(item.data.value, ToBeDetermined):
        result["value"] = item.data.value
    if item.shape is not None:
        shape_template = []
        for symbol in item.shape:
            if isinstance(symbol, tuple):  # variadic
                shape_template.append(f"{symbol[0]},...")
            else:
                shape_template.append(str(symbol))
        result["shape_template"] = shape_template

    elif isinstance(item.data.type, UnionType):
        result["type"] = [type_to_str(item) for item in item.data.type.__args__]
    else:
        result["type"] = [
            type_to_str(item.data.type),
        ]
    return result

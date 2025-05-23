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


import sys
from copy import deepcopy
from typing import Any

from mithril.framework.common import (
    NOT_GIVEN,
    ShapeTemplateType,
    Tensor,
)
from mithril.framework.logical.base import BaseKey
from mithril.models import (
    MLP,
    TBD,
    Add,
    BaseModel,
    Buffer,
    Connection,
    ConnectionType,
    Convolution1D,
    ExtendInfo,
    IOHyperEdge,
    Linear,
    MatrixMultiply,
    MaxPool1D,
    Model,
    PrimitiveUnion,
    Relu,
    Sigmoid,
    Sum,
)
from mithril.models.primitives import PrimitiveModel

from .test_utils import (
    get_all_data,
    get_all_metadata,
    get_all_nodes,
    get_all_reprs,
    get_all_uniadic_record,
    get_all_uniadics,
)


def get_all_variadics(model: BaseModel):
    return {repr.root for repr in get_all_reprs(model)} - {None}


def assert_objects_deleted(
    all_objects: set, current_objects: set, len_deleted_objects: int
):
    # find the deleted objects
    all_objects -= current_objects

    # also assert number of deleted objects vs expected number of deleted objects
    assert len(all_objects) == len_deleted_objects

    while all_objects:
        # Since getrefcount temporarily creates an additional ref,
        # also we have one additional ref in ref_var variables.
        # So refcount == 2 means it there is no additional reference left.
        deleted_obj = all_objects.pop()
        assert sys.getrefcount(deleted_obj) == 2


def test_deleted_variadic_ref_count_1() -> None:
    class TestModel(PrimitiveModel):
        input: Connection
        output: Connection

        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=BaseKey(shape=["a", "b", ("Var1", ...)], type=Tensor),
                output=BaseKey(shape=["c", "d", ("Var2", ...)], type=Tensor),
            )

    model = Model()
    submodel1 = TestModel()
    submodel2 = TestModel()

    assert submodel1.output.metadata.is_tensor
    assert submodel2.output.metadata.is_tensor
    assert submodel1.output.metadata.shape is not None
    assert submodel2.input.metadata.shape is not None
    ref_var1 = next(iter(submodel1.output.metadata.shape.reprs)).root
    ref_var2 = next(iter(submodel2.input.metadata.shape.reprs)).root

    model += submodel1
    model += submodel2
    # Since getrefcount temporarily creates an additional ref,
    # also we have one additional ref in ref_var variables.
    # So refcount == 2 means it there is no additional reference left.
    assert sys.getrefcount(ref_var1) == 3 and sys.getrefcount(ref_var2) == 2


def test_deleted_variadic_ref_count_2() -> None:
    model = Model()

    class MyModel(PrimitiveModel):
        input: Connection
        output: Connection

        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=BaseKey(shape=[("Var1", ...)], type=Tensor),
                output=BaseKey(shape=[("Var1", ...)], type=Tensor),
            )

    buff_model1 = MyModel()
    buff_model2 = MyModel()

    assert buff_model1.input.metadata.is_tensor
    assert buff_model2.input.metadata.is_tensor
    assert buff_model1.input.metadata.shape is not None
    assert buff_model2.output.metadata.shape is not None
    ref_var1 = next(iter(buff_model1.input.metadata.shape.reprs)).root
    ref_var2 = next(iter(buff_model2.output.metadata.shape.reprs)).root

    model += buff_model1
    model += buff_model2

    # memo =  {}
    # copied_model = deepcopy(model, memo)
    # assert not (id(ref_var1) in memo and id(ref_var2) in memo)

    assert (sys.getrefcount(ref_var1) == 2 and sys.getrefcount(ref_var2) == 3) or (
        sys.getrefcount(ref_var1) == 3 and sys.getrefcount(ref_var2) == 2
    )


def test_deleted_variadic_ref_count_3():
    # Gather all ever-existed variadics in the model that will be constructed
    all_variadics = set()
    all_variadics |= get_all_variadics(relu1 := Relu())
    all_variadics |= get_all_variadics(relu2 := Relu())
    all_variadics |= get_all_variadics(relu3 := Relu())
    all_variadics |= get_all_variadics(relu4 := Relu())

    model = Model()
    model += relu1
    model += relu2
    model += relu3
    model += relu4

    current_variadcs = get_all_variadics(model)
    assert_objects_deleted(all_variadics, current_variadcs, 3)


def test_deleted_variadic_ref_count_4():
    all_variadics = set()
    all_variadics |= get_all_variadics(sum1 := Sum())
    all_variadics |= get_all_variadics(sum2 := Sum())
    all_variadics |= get_all_variadics(sum3 := Sum())

    model = Model()
    model += sum1
    model += sum2
    model += sum3
    current_variadcs = get_all_variadics(model)
    assert_objects_deleted(all_variadics, current_variadcs, 2)


def test_deleted_variadic_ref_count_5():
    all_variadics = set()
    all_variadics |= get_all_variadics(lin_model1 := Linear())
    all_variadics |= get_all_variadics(relu1 := Relu())
    all_variadics |= get_all_variadics(lin_model2 := Linear())
    all_variadics |= get_all_variadics(relu2 := Relu())
    all_variadics |= get_all_variadics(lin_model3 := Linear())
    all_variadics |= get_all_variadics(matmul1 := MatrixMultiply())
    add1 = Add()
    add1.set_types(left=Tensor, right=Tensor)
    add1.set_cin("left")
    all_variadics |= get_all_variadics(add1)

    model = Model()
    model += lin_model1
    model += relu1
    model += lin_model2
    model += relu2
    model += lin_model3
    model += matmul1
    model += add1

    current_variadcs = get_all_variadics(model)
    assert_objects_deleted(all_variadics, current_variadcs, 9)


def test_deleted_variadic_ref_count_6():
    all_variadics = set()
    all_variadics |= get_all_variadics(
        conv1 := Convolution1D(kernel_size=2, out_channels=2)
    )
    all_variadics |= get_all_variadics(
        conv2 := Convolution1D(kernel_size=2, out_channels=2)
    )
    all_variadics |= get_all_variadics(maxpool1 := MaxPool1D(kernel_size=2))
    all_variadics |= get_all_variadics(
        conv3 := Convolution1D(kernel_size=2, out_channels=2)
    )
    all_variadics |= get_all_variadics(maxpool2 := MaxPool1D(kernel_size=2))

    model = Model()
    model += conv1
    model += conv2
    model += maxpool1
    model += conv3
    model += maxpool2

    current_variadics = get_all_variadics(model)

    assert_objects_deleted(all_variadics, current_variadics, 2)


def test_deleted_variadic_ref_count_7():
    all_variadics = set()
    add_1 = Add()
    add_1.set_types(left=Tensor, right=Tensor)
    add_2 = Add()
    add_2.set_types(left=Tensor, right=Tensor)
    add_3 = Add()
    add_3.set_types(left=Tensor, right=Tensor)
    add_4 = Add()
    add_4.set_types(left=Tensor, right=Tensor)
    add_5 = Add()
    add_5.set_types(left=Tensor, right=Tensor)
    add_6 = Add()
    add_6.set_types(left=Tensor, right=Tensor)
    all_variadics |= get_all_variadics(add_1)
    all_variadics |= get_all_variadics(add_2)
    all_variadics |= get_all_variadics(add_3)
    all_variadics |= get_all_variadics(add_4)
    all_variadics |= get_all_variadics(add_5)
    all_variadics |= get_all_variadics(add_6)

    model = Model()
    model |= add_1
    model |= add_2
    model |= add_3
    model |= add_4
    model |= add_5

    model.merge_connections(
        add_1.left, add_1.right, add_2.left, add_2.right, add_3.left, add_3.right
    )

    model |= add_6.connect(left=add_1.left, right="right", output="output")

    current_variadics = get_all_variadics(model)
    assert_objects_deleted(all_variadics, current_variadics, 9)


def test_deleted_variadic_ref_count_8():
    all_variadics = set()
    add1 = Add()
    add1.set_types(left=Tensor, right=Tensor)
    add1.set_cin("left")
    add2 = Add()
    add2.set_types(left=Tensor, right=Tensor)
    add2.set_cin("left")
    all_variadics |= get_all_variadics(add1)
    all_variadics |= get_all_variadics(add2)
    model1 = Model()
    model2 = Model()
    model1 += add1
    model2 += add2
    model1 += model2

    current_variadics = get_all_variadics(model1)

    assert_objects_deleted(all_variadics, current_variadics, 1)


def test_deleted_variadic_ref_count_9():
    all_variadics = set()
    add1 = Add()
    add1.set_types(left=Tensor, right=Tensor)
    add1.set_cin("left")
    all_variadics |= get_all_variadics(add1)

    model = Model()
    model += add1
    for _ in range(5):
        all_variadics |= get_all_variadics(model1 := deepcopy(model))
        model += model1

    current_variadics = get_all_variadics(model)

    assert_objects_deleted(all_variadics, current_variadics, 5)


def test_deleted_variadic_ref_count_10():
    all_variadics = set()
    buffer1 = Buffer()
    buffer1.set_types(input=Tensor)
    buffer2 = Buffer()
    buffer2.set_types(input=Tensor)
    buffer3 = Buffer()
    buffer3.set_types(input=Tensor)
    buffer4 = Buffer()
    buffer4.set_types(input=Tensor)
    all_variadics |= get_all_variadics(buffer1)
    all_variadics |= get_all_variadics(buffer2)
    all_variadics |= get_all_variadics(buffer3)
    all_variadics |= get_all_variadics(buffer4)

    model = Model()

    model += buffer1
    model += buffer2
    model += buffer3
    model += buffer4

    buffer1.set_shapes(input=[1, 2, 3])
    current_variadics = get_all_variadics(model)

    assert_objects_deleted(all_variadics, current_variadics, 4)


def test_deleted_uniadic_ref_count_3():
    all_uniadics = set()
    add_model = Add()
    add_model.set_types(left=Tensor, right=Tensor)
    add_model.set_shapes(left=["a", "b", "c", "d"])
    all_uniadics |= get_all_uniadics(add_model)

    add_model.set_shapes(left=["a", "a", "a", "a"])
    current_uniadics = get_all_uniadics(add_model)

    all_uniadics -= current_uniadics

    while all_uniadics:
        deleted_uniadic = all_uniadics.pop()
        assert sys.getrefcount(deleted_uniadic) == 2


def test_deleted_uniadic_ref_count_4():
    model = Model()
    buff1 = Buffer()
    buff1.set_types(input=Tensor)
    model += buff1
    model += Buffer()
    model += Buffer()
    model += Buffer()
    model += Buffer()

    buff1.set_shapes(input=["a", "b", "c", "d", "e", "f"])
    all_uniadics = get_all_uniadics(model)
    buff1.set_shapes(input=["a", "a", "a", "b", "b", "b"])
    current_uniadics = get_all_uniadics(model)

    all_uniadics -= current_uniadics

    while all_uniadics:
        deleted_uniadic = all_uniadics.pop()
        assert sys.getrefcount(deleted_uniadic) == 2


def test_deleted_uniadic_ref_count_5():
    class MyModel(PrimitiveModel):
        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=BaseKey(shape=["a", "b"], type=Tensor),
                output=BaseKey(shape=["c", "d"], type=Tensor),
            )

    all_uniadics = set()
    all_uniadics |= get_all_uniadics(tm1 := MyModel())
    all_uniadics |= get_all_uniadics(tm2 := MyModel())
    all_uniadics |= get_all_uniadics(tm3 := MyModel())
    all_uniadics |= get_all_uniadics(tm4 := MyModel())

    model = Model()
    model += tm1
    model += tm2
    model += tm3
    model += tm4

    current_uniadics = get_all_uniadics(model)
    assert_objects_deleted(all_uniadics, current_uniadics, 6)


def test_deleted_uniadic_ref_count_6():
    buff_model = Buffer()
    buff_model.set_shapes(input=["a", "b"])
    all_uniadics = get_all_uniadics(buff_model)
    buff_model.set_shapes(input=["a", "a"])
    current_uniadics = get_all_uniadics(buff_model)

    all_uniadics -= current_uniadics

    while all_uniadics:
        deleted_uniadic = all_uniadics.pop()
        assert sys.getrefcount(deleted_uniadic) == 2


def test_deleted_uniadic_ref_count_7():
    class MyModel(PrimitiveModel):
        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=BaseKey(shape=["a", "b", "c"], type=Tensor),
                output=BaseKey(shape=["c", "d", "e"], type=Tensor),
            )

    all_uniadics = set()
    all_uniadics |= get_all_uniadics(tm1 := MyModel())
    all_uniadics |= get_all_uniadics(tm2 := MyModel())
    all_uniadics |= get_all_uniadics(tm3 := MyModel())
    all_uniadics |= get_all_uniadics(tm4 := MyModel())

    model = Model()
    model += tm1
    model += tm2
    model += tm3
    model += tm4

    current_uniadics = get_all_uniadics(model)
    assert_objects_deleted(all_uniadics, current_uniadics, 9)


def test_deleted_uniadic_ref_count_8():
    class MyModel(PrimitiveModel):
        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=BaseKey(shape=["a", "b", "c"], type=Tensor),
                output=BaseKey(shape=["d", "e", "f"], type=Tensor),
            )

        def connect(  # type: ignore[override]
            self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
        ):
            return ExtendInfo(self, {"input": input, "output": output})

    all_uniadics = set()
    all_uniadics |= get_all_uniadics(tm1 := MyModel())
    all_uniadics |= get_all_uniadics(tm2 := MyModel())
    all_uniadics |= get_all_uniadics(tm3 := MyModel())
    all_uniadics |= get_all_uniadics(tm4 := MyModel())

    model = Model()
    model += tm1.connect(input="input", output="output")
    model += tm2
    model += tm3
    model += tm4
    model.set_shapes(input=[1, 2, 3])

    current_uniadics = get_all_uniadics(model)
    assert_objects_deleted(all_uniadics, current_uniadics, 12)


def test_deleted_uniadic_ref_count_9():
    class MyModel(PrimitiveModel):
        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=BaseKey(shape=[1, 1, 1], type=Tensor),
                output=BaseKey(shape=[1, 1, 1], type=Tensor),
            )

        def connect(  # type: ignore[override]
            self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
        ):
            return ExtendInfo(self, {"input": input, "output": output})

    all_uniadics = set()
    all_uniadics |= get_all_uniadics(tm1 := MyModel())
    all_uniadics |= get_all_uniadics(tm2 := MyModel())
    all_uniadics |= get_all_uniadics(tm3 := MyModel())
    all_uniadics |= get_all_uniadics(tm4 := MyModel())

    model = Model()
    model += tm1
    model += tm2
    model += tm3
    model += tm4

    current_uniadics = get_all_uniadics(model)
    assert_objects_deleted(all_uniadics, current_uniadics, 3)


def test_deleted_repr_ref_count_1():
    all_reprs = set()
    buffer1 = Buffer()
    buffer1.set_types(input=Tensor)
    buffer2 = Buffer()
    buffer2.set_types(input=Tensor)
    buffer3 = Buffer()
    buffer3.set_types(input=Tensor)
    buffer4 = Buffer()
    buffer4.set_types(input=Tensor)

    all_reprs |= get_all_reprs(buffer1)
    all_reprs |= get_all_reprs(buffer2)
    all_reprs |= get_all_reprs(buffer3)
    all_reprs |= get_all_reprs(buffer4)

    model = Model()
    model += buffer1
    model += buffer2
    model += buffer3
    model += buffer4

    current_reprs = get_all_reprs(model)
    assert buffer1.input.metadata.shape is not None
    print(sys.getrefcount(list(buffer1.input.metadata.shape.reprs)[0]))
    assert_objects_deleted(all_reprs, current_reprs, 3)


def test_deleted_repr_ref_count_2():
    all_reprs = set()
    buffer1 = Buffer()
    buffer1.set_types(input=Tensor)
    buffer2 = Buffer()
    buffer2.set_types(input=Tensor)
    buffer3 = Buffer()
    buffer3.set_types(input=Tensor)
    buffer4 = Buffer()
    buffer4.set_types(input=Tensor)
    all_reprs |= get_all_reprs(buffer1)
    all_reprs |= get_all_reprs(buffer2)
    all_reprs |= get_all_reprs(buffer3)
    all_reprs |= get_all_reprs(buffer4)

    model = Model()
    model += buffer1
    model += buffer2
    model += buffer3
    model += buffer4

    buffer1_shape: ShapeTemplateType = ["a", "b", ("Var1", ...)]
    buffer1.set_shapes(input=buffer1_shape)
    all_reprs |= get_all_reprs(model)
    buffer1_shape = ["a", ("Var1", ...), "b"]
    buffer1.set_shapes(input=buffer1_shape)
    all_reprs |= get_all_reprs(model)
    buffer1_shape = [("Var1", ...), "a", "b"]
    buffer1.set_shapes(input=buffer1_shape)
    all_reprs |= get_all_reprs(model)
    buffer1.set_shapes(input=["a", "b"])
    current_reprs = get_all_reprs(model)

    assert_objects_deleted(all_reprs, current_reprs, 5)


def test_deleted_repr_ref_count_3():
    all_reprs = set()
    buffer1 = Buffer()
    buffer1.set_types(input=Tensor)
    buffer2 = Buffer()
    buffer2.set_types(input=Tensor)
    buffer3 = Buffer()
    buffer3.set_types(input=Tensor)
    buffer4 = Buffer()
    buffer4.set_types(input=Tensor)
    all_reprs |= get_all_reprs(buffer1)
    all_reprs |= get_all_reprs(buffer2)
    all_reprs |= get_all_reprs(buffer3)
    all_reprs |= get_all_reprs(buffer4)

    model = Model()
    model += buffer1
    model += buffer2
    model += buffer3
    model += buffer4

    buffer1_shape: ShapeTemplateType = ["a", "b", ("Var1", ...)]
    buffer1.set_shapes(input=buffer1_shape)
    all_reprs |= get_all_reprs(model)
    buffer1_shape = ["a", ("Var1", ...), "b"]
    buffer1.set_shapes(input=buffer1_shape)
    all_reprs |= get_all_reprs(model)
    buffer1_shape = [("Var1", ...), "a", "b"]
    buffer1.set_shapes(input=buffer1_shape)
    all_reprs |= get_all_reprs(model)
    current_reprs = get_all_reprs(model)
    assert_objects_deleted(all_reprs, current_reprs, 3)


def test_deleted_repr_ref_count_4():
    all_reprs = set()
    buffer1 = Buffer()
    buffer1.set_types(input=Tensor)
    buffer2 = Buffer()
    buffer2.set_types(input=Tensor)
    buffer3 = Buffer()
    buffer3.set_types(input=Tensor)
    buffer4 = Buffer()
    buffer4.set_types(input=Tensor)
    all_reprs |= get_all_reprs(buffer1)
    all_reprs |= get_all_reprs(buffer2)
    all_reprs |= get_all_reprs(buffer3)
    all_reprs |= get_all_reprs(buffer4)

    buffer1.set_shapes(input=[1, 1])
    all_reprs |= get_all_reprs(buffer1)

    model = Model()
    model += buffer1
    model += buffer2
    model += buffer3
    model += buffer4

    current_reprs = get_all_reprs(model)
    assert_objects_deleted(all_reprs, current_reprs, 3)


def test_deleted_repr_ref_count_5() -> None:
    class MyModel(PrimitiveModel):
        input: Connection
        output: Connection

        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=BaseKey(shape=[("Var1", ...), "a"], type=Tensor),
                output=BaseKey(shape=["a", ("Var1", ...)], type=Tensor),
            )

        def connect(  # type: ignore[override]
            self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
        ):
            return ExtendInfo(self, {"input": input, "output": output})

    all_reprs = set()
    all_reprs |= get_all_reprs(tm1 := MyModel())
    all_reprs |= get_all_reprs(tm2 := MyModel())
    all_reprs |= get_all_reprs(tm3 := MyModel())
    all_reprs |= get_all_reprs(tm4 := MyModel())
    all_reprs |= get_all_reprs(tm5 := MyModel())
    all_reprs |= get_all_reprs(tm6 := MyModel())
    all_reprs |= get_all_reprs(tm7 := MyModel())
    all_reprs |= get_all_reprs(tm8 := MyModel())
    all_reprs |= get_all_reprs(tm9 := MyModel())

    model = Model()
    model |= tm1.connect(input="input1")
    model |= tm2.connect(input=tm1.output)
    model |= tm3.connect(input=tm2.output, output="output1")

    model |= tm4.connect(input="input2")
    model |= tm5.connect(input=tm1.output)
    model |= tm6.connect(input=tm2.output, output="output2")

    model |= tm7.connect(input="input3")
    model |= tm8.connect(input=tm1.output)
    model |= tm9.connect(input=tm2.output, output="output3")

    model.set_shapes(
        input1=["a", "b", "c"],
        input2=["c", "a", "b"],
        input3=["b", "c", "a"],
    )

    current_reprs = get_all_reprs(model)
    assert_objects_deleted(all_reprs, current_reprs, 15)


# @pytest.mark.skip("investigate later")
def test_deleted_repr_ref_count_6() -> None:
    class MyModel(PrimitiveModel):
        input: Connection
        output: Connection

        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=BaseKey(shape=[("Var1", ...), "a"], type=Tensor),
                output=BaseKey(shape=["a", ("Var1", ...)], type=Tensor),
            )

        def connect(  # type: ignore[override]
            self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
        ):
            return ExtendInfo(self, {"input": input, "output": output})

    all_reprs = set()
    all_reprs |= get_all_reprs(tm1 := MyModel())
    all_reprs |= get_all_reprs(tm2 := MyModel())
    all_reprs |= get_all_reprs(tm3 := MyModel())
    all_reprs |= get_all_reprs(tm4 := MyModel())
    all_reprs |= get_all_reprs(tm5 := MyModel())
    all_reprs |= get_all_reprs(tm6 := MyModel())
    all_reprs |= get_all_reprs(tm7 := MyModel())
    all_reprs |= get_all_reprs(tm8 := MyModel())
    all_reprs |= get_all_reprs(tm9 := MyModel())

    model = Model()
    model |= tm1.connect(input="input1")
    model |= tm2.connect(input=tm1.output)
    model |= tm3.connect(input=tm2.output, output="output1")

    model |= tm4.connect(input="input2")
    model |= tm5.connect(input=tm1.output)
    model |= tm6.connect(input=tm2.output, output="output2")

    model |= tm7.connect(input="input3")
    model |= tm8.connect(input=tm1.output)
    model |= tm9.connect(input=tm2.output, output="output3")

    model.set_shapes(
        input1=["a", "b", "c"],
        input2=["c", "a", "b"],
        input3=["b", "c", "a"],
    )

    current_reprs = get_all_reprs(model)
    assert_objects_deleted(all_reprs, current_reprs, 15)


def test_deleted_repr_ref_count_7() -> None:
    class MyModel(PrimitiveModel):
        input: Connection
        output: Connection

        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=BaseKey(shape=[("Var1", ...), "a"], type=Tensor),
                output=BaseKey(shape=["a", ("Var1", ...)], type=Tensor),
            )

        def connect(  # type: ignore[override]
            self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
        ):
            return ExtendInfo(self, {"input": input, "output": output})

    all_reprs = set()
    all_reprs |= get_all_reprs(tm1 := MyModel())
    all_reprs |= get_all_reprs(tm2 := MyModel())
    all_reprs |= get_all_reprs(tm3 := MyModel())
    all_reprs |= get_all_reprs(tm4 := MyModel())

    model = Model()
    model |= tm1.connect(input="input1")
    model |= tm2.connect(input=tm1.output, output="output")

    model |= tm3.connect(input="input2")
    model |= tm4.connect(input=tm1.output, output="output2")

    model.set_shapes(input1=["a", "b"], input2=["b", "a"])

    current_reprs = get_all_reprs(model)
    assert_objects_deleted(all_reprs, current_reprs, 6)


def test_deleted_repr_ref_count_8() -> None:
    class MyModel(PrimitiveModel):
        input: Connection
        output: Connection

        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=BaseKey(shape=["a", "b"], type=Tensor),
                output=BaseKey(shape=["b", "a"], type=Tensor),
            )

        def connect(  # type: ignore[override]
            self, input: ConnectionType = NOT_GIVEN, output: ConnectionType = NOT_GIVEN
        ):
            return ExtendInfo(self, {"input": input, "output": output})

    all_reprs = set()
    all_reprs |= get_all_reprs(tm1 := MyModel())
    all_reprs |= get_all_reprs(tm2 := MyModel())
    all_reprs |= get_all_reprs(tm3 := MyModel())
    all_reprs |= get_all_reprs(tm4 := MyModel())

    model = Model()
    model |= tm1.connect(input="input1")
    model |= tm2.connect(input=tm1.output, output="output")

    model |= tm3.connect(input="input2")
    model |= tm4.connect(input=tm1.output, output="output2")

    model.set_shapes(input1=["a", "b"], input2=["b", "a"])

    current_reprs = get_all_reprs(model)
    assert_objects_deleted(all_reprs, current_reprs, 6)


def test_deleted_repr_ref_count_9():
    all_reprs = set()
    buffer1 = Buffer()
    buffer1.set_types(input=Tensor)
    buffer2 = Buffer()
    buffer2.set_types(input=Tensor)
    all_reprs |= get_all_reprs(buffer1)
    all_reprs |= get_all_reprs(buffer2)

    model = Model()
    model |= buffer1.connect(input="input1", output="output1")
    model |= buffer2.connect(input="input2", output="output2")

    model.set_shapes(input1=["a", "b"], input2=["a", "b"])

    current_reprs = get_all_reprs(model)
    assert_objects_deleted(all_reprs, current_reprs, 1)


def test_deleted_repr_ref_count_10():
    all_reprs = set()
    buffer1 = Buffer()
    buffer1.set_types(input=Tensor)
    buffer2 = Buffer()
    buffer2.set_types(input=Tensor)
    all_reprs |= get_all_reprs(buffer1)
    all_reprs |= get_all_reprs(buffer2)

    model = Model()

    model |= buffer1.connect(input="input1", output="output1")
    model |= buffer2.connect(input="output1", output="output2")

    current_reprs = get_all_reprs(model)
    assert_objects_deleted(all_reprs, current_reprs, 1)


def test_deleted_repr_ref_count_10_1():
    all_reprs = set()
    buffer1 = Buffer()
    buffer1.set_types(input=Tensor)
    buffer2 = Buffer()
    buffer2.set_types(input=Tensor)
    all_reprs |= get_all_reprs(buffer1)
    all_reprs |= get_all_reprs(buffer2)

    model = Model()

    model |= buffer1.connect(input="input1", output="output1")
    model |= buffer2.connect(input="input2", output="output2")

    model.set_shapes(input1=[1, 2], input2=[1, 2])

    current_reprs = get_all_reprs(model)
    assert_objects_deleted(all_reprs, current_reprs, 1)


def test_deleted_node_ref_count_1():
    all_reprs = set()
    buffer1 = Buffer()
    buffer1.set_types(input=Tensor)
    buffer2 = Buffer()
    buffer2.set_types(input=Tensor)
    buffer3 = Buffer()
    buffer3.set_types(input=Tensor)
    all_reprs |= get_all_nodes(buffer1)
    all_reprs |= get_all_nodes(buffer2)
    all_reprs |= get_all_nodes(buffer3)

    model = Model()

    model += buffer1
    model += buffer2
    model += buffer3

    current_reprs = get_all_nodes(model)
    assert_objects_deleted(all_reprs, current_reprs, 2)


def test_deleted_node_ref_count_2():
    all_reprs = set()
    buffer1 = Buffer()
    buffer1.set_types(input=Tensor)
    buffer2 = Buffer()
    buffer2.set_types(input=Tensor)
    buffer3 = Buffer()
    buffer3.set_types(input=Tensor)
    all_reprs |= get_all_nodes(buffer1)
    all_reprs |= get_all_nodes(buffer2)
    all_reprs |= get_all_nodes(buffer3)

    model = Model()

    model += buffer1
    model += buffer2
    model += buffer3

    buffer3_shape: ShapeTemplateType = ["a", ("Var1", ...)]
    buffer3.set_shapes(input=buffer3_shape)
    buffer2_shape: ShapeTemplateType = [("Var1", ...), "a"]
    buffer2.set_shapes(output=buffer2_shape)

    current_reprs = get_all_nodes(model)
    assert_objects_deleted(all_reprs, current_reprs, 2)


def test_deleted_node_ref_count_3():
    all_reprs = set()
    add1 = Add()
    add1.set_types(left=Tensor, right=Tensor)
    add1.set_cin("left")
    add2 = Add()
    add2.set_types(left=Tensor, right=Tensor)
    add2.set_cin("left")
    add3 = Add()
    add3.set_types(left=Tensor, right=Tensor)
    add3.set_cin("left")
    all_reprs |= get_all_nodes(add1)
    all_reprs |= get_all_nodes(add2)
    all_reprs |= get_all_nodes(add3)

    model = Model()

    model += add1
    model += add2
    model += add3

    current_reprs = get_all_nodes(model)
    assert_objects_deleted(all_reprs, current_reprs, 2)


def test_deleted_node_ref_count_4():
    all_reprs = set()
    buffer1 = Buffer()
    buffer1.set_types(input=Tensor)
    buffer2 = Buffer()
    buffer2.set_types(input=Tensor)
    buffer3 = Buffer()
    buffer3.set_types(input=Tensor)
    buffer4 = Buffer()
    buffer4.set_types(input=Tensor)
    all_reprs |= get_all_nodes(buffer1)
    all_reprs |= get_all_nodes(buffer2)
    all_reprs |= get_all_nodes(buffer3)
    all_reprs |= get_all_nodes(buffer4)

    model = Model()

    model |= buffer1.connect(input="input1")
    model |= buffer2.connect(input=buffer1.output, output="output1")

    model |= buffer3.connect(input="input2")
    model |= buffer4.connect(input=buffer1.output, output="output2")

    input_shape: ShapeTemplateType = ["a", ("Var1", ...)]
    model.set_shapes(input1=input_shape, input2=input_shape)

    current_reprs = get_all_nodes(model)
    assert_objects_deleted(all_reprs, current_reprs, 3)


def test_deleted_tensors_ref_count_1():
    all_reprs = set()
    all_reprs |= get_all_data(buffer1 := Buffer())
    all_reprs |= get_all_data(buffer2 := Buffer())
    all_reprs |= get_all_data(buffer3 := Buffer())
    all_reprs |= get_all_data(buffer4 := Buffer())

    model = Model()

    model += buffer1
    model += buffer2
    model += buffer3
    model += buffer4

    current_reprs = get_all_data(model)
    assert_objects_deleted(all_reprs, current_reprs, 3)


def test_deleted_tensors_ref_count_2():
    all_reprs = set()
    all_reprs |= get_all_data(buffer1 := Buffer())
    all_reprs |= get_all_data(buffer2 := Buffer())
    all_reprs |= get_all_data(buffer3 := Buffer())
    all_reprs |= get_all_data(buffer4 := Buffer())

    model = Model()

    model |= buffer1.connect(input="input1")
    model |= buffer2.connect(input=buffer1.output, output="output1")

    model |= buffer3.connect(input="input2")
    model |= buffer4.connect(input=buffer3.output, output="output2")

    current_reprs = get_all_data(model)
    assert_objects_deleted(all_reprs, current_reprs, 2)


def test_deleted_tensors_ref_count_3():
    all_reprs = set()
    all_reprs |= get_all_data(buffer1 := Buffer())
    all_reprs |= get_all_data(buffer2 := Buffer())
    all_reprs |= get_all_data(buffer3 := Buffer())
    all_reprs |= get_all_data(buffer4 := Buffer())
    all_reprs |= get_all_data(buffer5 := Buffer())
    all_reprs |= get_all_data(buffer6 := Buffer())
    all_reprs |= get_all_data(buffer7 := Buffer())

    model = Model()
    model |= buffer1
    model |= buffer2
    model |= buffer3
    model |= buffer4.connect(input="input4")
    model |= buffer5.connect(input="input5")
    model |= buffer6.connect(input="input6")
    model.merge_connections(buffer1.input, buffer2.input, buffer3.input, buffer4.output)
    model |= buffer7.connect(input=buffer1.input)
    model.expose_keys(
        output1=buffer1.output,
        output2=buffer2.output,
        output3=buffer3.output,
        output4=buffer4.output,
        output5=buffer5.output,
        output6=buffer6.output,
        output7=buffer7.output,
    )
    current_reprs = get_all_data(model)
    # NOTE: 7 output tensors are exposed so created and they replaced the previous ones.
    # We have to take this account while checking the deleted objects. We expect 4
    # objects to be deleted but after |= operation, we will have 7 additional objects
    # in the set. So we should expect 11 objects to be deleted.
    all_reprs |= current_reprs
    # assert_objects_deleted(all_reprs, current_reprs, 4 + 7)
    assert_objects_deleted(all_reprs, current_reprs, 4)


def test_deleted_scalars_ref_count_1():
    all_reprs = set()

    all_reprs |= get_all_data(union1 := PrimitiveUnion(n=1))
    all_reprs |= get_all_data(union2 := PrimitiveUnion(n=1))
    all_reprs |= get_all_data(union3 := PrimitiveUnion(n=1))

    model = Model()

    model += union1
    model += union2
    model += union3

    current_reprs = get_all_data(model)
    assert_objects_deleted(all_reprs, current_reprs, 2)


def test_deleted_edge_ref_count_1():
    all_metadata = set()

    all_metadata |= get_all_metadata(buffer1 := Buffer())
    all_metadata |= get_all_metadata(buffer2 := Buffer())
    all_metadata |= get_all_metadata(buffer3 := Buffer())

    model = Model()

    model += buffer1
    model += buffer2
    model += buffer3

    current_metadata = get_all_metadata(model)
    assert_objects_deleted(all_metadata, current_metadata, 2)


def test_deleted_edge_ref_count_2():
    all_metadata = set()

    all_metadata |= get_all_metadata(add1 := Add())
    all_metadata |= get_all_metadata(add2 := Add())
    all_metadata |= get_all_metadata(add3 := Add())

    model = Model()
    add1.set_cin("left")
    add2.set_cin("left")
    add3.set_cin("left")

    model += add1
    model += add2
    model += add3

    current_metadata = get_all_metadata(model)
    assert_objects_deleted(all_metadata, current_metadata, 2)


def test_deleted_edge_ref_count_3():
    all_metadata = set()

    all_metadata |= get_all_metadata(add1 := Add())
    all_metadata |= get_all_metadata(add2 := Add())
    all_metadata |= get_all_metadata(add3 := Add())

    model = Model()

    model |= add1.connect(output="output", right="right3")
    model |= add2.connect(output=add1.left, right="right2")
    model |= add3.connect(output=add2.left, right="right1", left="left")

    current_metadata = get_all_metadata(model)
    assert_objects_deleted(all_metadata, current_metadata, 2)


def test_deleted_edge_ref_count_4():
    all_metadata = set()
    all_metadata |= get_all_metadata(add1 := Add())
    all_metadata |= get_all_metadata(add2 := Add())
    all_metadata |= get_all_metadata(add3 := Add())
    all_metadata |= get_all_metadata(add4 := Add())

    add1.set_cin("left")
    add2.set_cin("left")
    add3.set_cin("left")
    add4.set_cin("left")

    model4 = Model()
    model3 = Model()
    model2 = Model()
    model1 = Model()

    model4 += add1

    model3 += model4
    model3 += add2

    model2 += model3
    model2 += add3

    model1 += model2
    model1 += add4

    current_metadata = get_all_metadata(model1)

    assert_objects_deleted(all_metadata, current_metadata, 3)


def test_deleted_edge_ref_count_5():
    all_metadata = set()
    all_metadata |= get_all_metadata(add1 := Add())
    all_metadata |= get_all_metadata(add2 := Add())
    all_metadata |= get_all_metadata(add3 := Add())
    all_metadata |= get_all_metadata(add4 := Add())

    add1.set_cin("left")
    add2.set_cin("left")
    add3.set_cin("left")
    add4.set_cin("left")

    model4 = Model()
    model3 = Model()
    model2 = Model()
    model1 = Model()

    model4 += add1

    model3 += model4
    model3 += add2

    model2 += model3
    model2 += add3

    model1 += model2
    model1 += add4

    current_metadata = get_all_metadata(model1)

    assert_objects_deleted(all_metadata, current_metadata, 3)


# def test_deleted_edge_ref_count_6():
#     all_metadata = set()
#     all_metadata |= get_all_metadata(sigmoid1 := Sigmoid())
#     all_metadata |= get_all_metadata(sigmoid2 := Sigmoid())
#     all_metadata |= get_all_metadata(sigmoid3 := Sigmoid())
#     all_metadata |= get_all_metadata(sigmoid4 := Sigmoid())

#     three_sigmoid_model = Model()

#     three_sigmoid_model += sigmoid1(input="input1", output="output1")
#     three_sigmoid_model += sigmoid2(input="input2", output="output2")
#     three_sigmoid_model += sigmoid3(input="input3", output="output3")

#     main_model = Model()

#     main_model += three_sigmoid_model(
#         input1="input1",
#         input2="input2",
#         input3="input3",
#         output1="output1",
#         output2="output2",
#         output3="output3",
#     )
#     conn = Connect(main_model.output1, main_model.input2, name="abcd")

#     main_model += sigmoid4(input=conn, output="output5")

#     current_metadata = get_all_metadata(main_model)

#     assert_objects_deleted(all_metadata, current_metadata, 2


def test_deleted_edge_ref_count_6():
    all_metadata = set()
    all_metadata |= get_all_metadata(sigmoid1 := Sigmoid())
    all_metadata |= get_all_metadata(sigmoid2 := Sigmoid())
    all_metadata |= get_all_metadata(sigmoid3 := Sigmoid())
    all_metadata |= get_all_metadata(sigmoid4 := Sigmoid())

    three_sigmoid_model = Model()

    three_sigmoid_model |= sigmoid1.connect(input="input1")
    three_sigmoid_model |= sigmoid2.connect(input="input2")
    three_sigmoid_model |= sigmoid3.connect(input="input3")
    three_sigmoid_model.expose_keys(
        output1=sigmoid1.output, output2=sigmoid2.output, output3=sigmoid3.output
    )

    main_model = Model()
    main_model |= three_sigmoid_model.connect(
        input1="input1", input2="input2", input3="input3"
    )
    main_model.merge_connections(sigmoid1.output, sigmoid2.input, name="abcd")
    main_model |= sigmoid4.connect(input=main_model.abcd)  # type: ignore
    main_model.expose_keys(
        output1=sigmoid1.output,
        output2=sigmoid2.output,
        output3=sigmoid3.output,
        output5=sigmoid4.output,
    )

    current_metadata = get_all_metadata(main_model)  # 2input + 4output = 6
    # NOTE: 4 new output metadata is created, in order to take these into account
    # we should add current_metadata to all_metadata.
    all_metadata |= current_metadata  # 8old + 4new = 12

    assert_objects_deleted(all_metadata, current_metadata, 2)


def test_deleted_uni_record_ref_count_1():
    all_record = set()
    all_record |= get_all_uniadic_record(sigmoid1 := Sigmoid())
    sigmoid1.set_shapes(input=[2, 3, 4])
    all_record |= get_all_uniadic_record(sigmoid1)

    all_record |= get_all_uniadic_record(sigmoid2 := Sigmoid())
    sigmoid2.set_shapes(input=[2, 3, 4])
    all_record |= get_all_uniadic_record(sigmoid2)

    model = Model()

    model += sigmoid1
    model += sigmoid2

    current_records = get_all_uniadic_record(model)

    assert_objects_deleted(all_record, current_records, 3)


def test_deleted_uni_record_ref_count_2():
    all_record = set()
    all_record |= get_all_uniadic_record(sigmoid1 := Sigmoid())
    sigmoid1.set_shapes(input=[2, 3, 4])
    all_record |= get_all_uniadic_record(sigmoid1)

    all_record |= get_all_uniadic_record(sigmoid2 := Sigmoid())
    sigmoid2.set_shapes(input=[2, 3, 4])
    all_record |= get_all_uniadic_record(sigmoid2)

    all_record |= get_all_uniadic_record(sigmoid3 := Sigmoid())
    sigmoid3.set_shapes(input=[2, 3, 4])
    all_record |= get_all_uniadic_record(sigmoid3)

    all_record |= get_all_uniadic_record(sigmoid4 := Sigmoid())
    sigmoid4.set_shapes(input=[2, 3, 4])
    all_record |= get_all_uniadic_record(sigmoid4)

    model = Model()

    model += sigmoid1
    model += sigmoid2
    model += sigmoid3
    model += sigmoid4

    current_records = get_all_uniadic_record(model)

    assert_objects_deleted(all_record, current_records, 9)


def test_deleted_uni_record_ref_count_3():
    all_record = set()
    all_record |= get_all_uniadic_record(sigmoid1 := Sigmoid())

    sigmoid1_shape: ShapeTemplateType = ["a", "b", 4]
    sigmoid1.set_shapes(input=sigmoid1_shape)
    all_record |= get_all_uniadic_record(sigmoid1)

    sigmoid1_shape = [4, 4, 4]
    sigmoid1.set_shapes(input=sigmoid1_shape)
    all_record |= get_all_uniadic_record(sigmoid1)

    current_records = get_all_uniadic_record(sigmoid1)

    assert_objects_deleted(all_record, current_records, 2)


def test_deleted_uni_record_ref_count_4():
    all_record = set()
    all_record |= get_all_uniadic_record(sigmoid1 := Sigmoid())

    sigmoid1_shape: ShapeTemplateType = [1, ("V1", ...)]
    sigmoid1.set_shapes(input=sigmoid1_shape)
    all_record |= get_all_uniadic_record(sigmoid1)

    sigmoid1_shape = [("V1", ...), "a"]
    sigmoid1.set_shapes(input=sigmoid1_shape)
    all_record |= get_all_uniadic_record(sigmoid1)

    sigmoid1_shape = [("V1", ...), 1]
    sigmoid1.set_shapes(input=sigmoid1_shape)
    all_record |= get_all_uniadic_record(sigmoid1)

    current_records = get_all_uniadic_record(sigmoid1)

    assert_objects_deleted(all_record, current_records, 1)


def test_deleted_uni_record_ref_count_5():
    all_record = set()
    all_record |= get_all_uniadic_record(sigmoid1 := Sigmoid())

    sigmoid1_shape: ShapeTemplateType = ["b", ("V1", ...)]
    sigmoid1.set_shapes(input=sigmoid1_shape)
    all_record |= get_all_uniadic_record(sigmoid1)

    sigmoid1_shape = [("V1", ...), "a"]
    sigmoid1.set_shapes(input=sigmoid1_shape)
    all_record |= get_all_uniadic_record(sigmoid1)

    sigmoid1_shape = [1, ("V1", ...), 1]
    sigmoid1.set_shapes(input=sigmoid1_shape)

    current_records = get_all_uniadic_record(sigmoid1)

    assert_objects_deleted(all_record, current_records, 1)


def test_deleted_uniadic_ref_count_2() -> None:
    model = Model()

    class MyModel(PrimitiveModel):
        input: Connection
        output: Connection

        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=BaseKey(shape=["a1"], type=Tensor),
                output=BaseKey(shape=["a2"], type=Tensor),
            )

    buff_model1 = MyModel()
    buff_model2 = MyModel()

    assert buff_model1.output.metadata.is_tensor
    assert buff_model2.input.metadata.is_tensor
    assert buff_model1.output.metadata.shape is not None
    assert buff_model2.input.metadata.shape is not None
    ref_var1 = next(iter(buff_model1.output.metadata.shape.reprs))[0]
    ref_var2 = next(iter(buff_model2.input.metadata.shape.reprs))[0]

    model += buff_model1
    model += buff_model2

    diff_roots = set()

    for tensor in get_all_data(model):
        assert tensor.is_tensor
        node = tensor.shape
        assert node is not None
        for repr in node.reprs:
            diff_roots.add(repr.root)

    # Since getrefcount temporarily creates an additional ref,
    # also we have one additional ref in ref_var variables.
    # So refcount == 2 means it there is no additional reference left.
    assert sys.getrefcount(ref_var1) == 2 or sys.getrefcount(ref_var2) == 2


def test_deleted_uniadic_ref_count() -> None:
    class MyModel(PrimitiveModel):
        input: Connection
        output: Connection

        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=BaseKey(shape=["a", "b"], type=Tensor),
                output=BaseKey(shape=["c", "d"], type=Tensor),
            )

    model = Model()
    submodel1 = MyModel()
    submodel2 = MyModel()

    assert submodel1.output.metadata.is_tensor
    assert submodel2.input.metadata.is_tensor
    assert submodel1.output.metadata.shape is not None
    assert submodel2.input.metadata.shape is not None
    ref_var1 = next(iter(submodel1.output.metadata.shape.reprs))[0]
    ref_var2 = next(iter(submodel2.input.metadata.shape.reprs))[0]

    model += submodel1
    model += submodel2

    # Since getrefcount temporarily creates an additional ref,
    # also we have one additional ref in ref_var variables.
    # So refcount == 2 means it there is no additional reference left.
    assert sys.getrefcount(ref_var1) == 2 or sys.getrefcount(ref_var2) == 2


def test_deleted_repr_ref_count() -> None:
    class MyModel(PrimitiveModel):
        input: Connection
        output: Connection

        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=BaseKey(shape=["a", "b"], type=Tensor),
                output=BaseKey(shape=["c", "d"], type=Tensor),
            )

    model = Model()
    submodel1 = MyModel()
    submodel2 = MyModel()

    assert submodel1.output.metadata.is_tensor
    assert submodel2.input.metadata.is_tensor
    assert submodel1.output.metadata.shape is not None
    assert submodel2.input.metadata.shape is not None
    ref_var1 = next(iter(submodel1.output.metadata.shape.reprs))
    ref_var2 = next(iter(submodel2.input.metadata.shape.reprs))

    model += submodel1
    model += submodel2

    # Since getrefcount temporarily creates an additional ref,
    # also we have one additional ref in ref_var variables.
    # So refcount == 2 means it there is no additional reference left.
    assert sys.getrefcount(ref_var1) == 2 or sys.getrefcount(ref_var2) == 2


def test_deleted_node_ref_count() -> None:
    class MyModel(PrimitiveModel):
        input: Connection
        output: Connection

        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=BaseKey(shape=["a", "b"], type=Tensor),
                output=BaseKey(shape=["c", "d"], type=Tensor),
            )

    model = Model()
    submodel1 = MyModel()
    submodel2 = MyModel()

    assert submodel1.output.metadata.is_tensor
    assert submodel2.input.metadata.is_tensor
    ref_var1 = submodel1.output.metadata.shape
    ref_var2 = submodel2.input.metadata.shape

    model += submodel1
    model += submodel2

    # Since getrefcount temporarily creates an additional ref,
    # also we have one additional ref in ref_var variables.
    # So refcount == 2 means it there is no additional reference left.
    assert sys.getrefcount(ref_var1) == 2 or sys.getrefcount(ref_var2) == 2


def test_deleted_tensor_ref_count() -> None:
    class MyModel(PrimitiveModel):
        input: Connection
        output: Connection

        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=BaseKey(shape=["a", "b"], type=Tensor),
                output=BaseKey(shape=["c", "d"], type=Tensor),
            )

    model = Model()
    submodel1 = MyModel()
    submodel2 = MyModel()

    ref_var1 = submodel1.output.metadata
    ref_var2 = submodel2.input.metadata

    model += submodel1
    model += submodel2

    # Since getrefcount temporarily creates an additional ref,
    # also we have one additional ref in ref_var variables.
    # So refcount == 2 means it there is no additional reference left.
    assert sys.getrefcount(ref_var1) == 2 or sys.getrefcount(ref_var2) == 2


def test_deleted_edge_ref_count() -> None:
    class MyModel(PrimitiveModel):
        input: Connection
        output: Connection

        def __init__(self) -> None:
            super().__init__(
                formula_key="buffer",
                input=BaseKey(shape=["a", "b"], type=Tensor),
                output=BaseKey(shape=["c", "d"], type=Tensor),
            )

    model = Model()
    submodel1 = MyModel()
    submodel2 = MyModel()

    ref_var1 = submodel1.output.metadata
    ref_var2 = submodel2.input.metadata

    model += submodel1
    model += submodel2

    # Since getrefcount temporarily creates an additional ref,
    # also we have one additional ref in ref_var variables.
    # So refcount == 2 means it there is no additional reference left.
    assert sys.getrefcount(ref_var1) == 2 or sys.getrefcount(ref_var2) == 2


def test_deleted_dependency_map():
    model = Model()
    add_1 = Add()
    add_2 = Add()
    model |= add_1.connect(left="left1", right="right1", output="output1")

    assert add_1.dependency_map._global_input_dependency_map is None
    assert add_1.dependency_map._global_input_dependency_map_cache is None
    assert add_1.dependency_map._global_output_dependency_map is None
    assert add_1.dependency_map._global_output_dependency_map_cache is None
    assert add_1.dependency_map._local_input_dependency_map is None

    model |= add_2.connect(left="left2", right="right2", output="output2")

    assert add_2.dependency_map._global_input_dependency_map is None
    assert add_2.dependency_map._global_input_dependency_map_cache is None
    assert add_2.dependency_map._global_output_dependency_map is None
    assert add_2.dependency_map._global_output_dependency_map_cache is None
    assert add_2.dependency_map._local_input_dependency_map is None


def test_total_object_count_ten_layer_mlp():
    model = MLP(
        activations=[Relu() for _ in range(10)], dimensions=[10 for _ in range(10)]
    )
    memo: dict[Any, Any] = {}
    deepcopy(model, memo)
    assert len(memo) <= 9000


def test_hyperedges_match_list_of_tensors_with_tbd():
    t1: Tensor[int] = Tensor(type=int)
    t2: Tensor[int] = Tensor(type=int)
    t3: Tensor[int] = Tensor(type=int)
    t4: Tensor[int] = Tensor(type=int)

    edge1 = IOHyperEdge(value=[[t1, t2], [t3, t4]])
    edge2 = IOHyperEdge(value=[[t2, t3], TBD])

    edge1.match(edge2)

    assert sys.getrefcount(t2) == 2
    assert sys.getrefcount(t3) == 2


def test_hyperedges_match_list_of_tensors():
    t1: Tensor[int] = Tensor(type=int)
    t2: Tensor[int] = Tensor(type=int)
    t3: Tensor[int] = Tensor(type=int)
    t4: Tensor[int] = Tensor(type=int)
    t5: Tensor[int] = Tensor(type=int)

    edge1 = IOHyperEdge(value=[[t1, t2], [t3, t4]])
    edge2 = IOHyperEdge(value=[[t2, t3], [t4, t5]])

    edge1.match(edge2)

    assert sys.getrefcount(t1) != 2
    assert sys.getrefcount(t2) == 2
    assert sys.getrefcount(t3) == 2
    assert sys.getrefcount(t4) == 2
    assert sys.getrefcount(t5) == 2


def test_hyperedges_match_list_of_tensors_with_one_edge_free():
    t1: Tensor[int] = Tensor(type=int)
    t2: Tensor[int] = Tensor(type=int)
    t3: Tensor[int] = Tensor(type=int)
    t4: Tensor[int] = Tensor(type=int)
    t5: Tensor[int] = Tensor(type=int)

    edge1 = IOHyperEdge(value=[[t1, t2], [t3, t4]])
    edge2 = IOHyperEdge(value=[[t2, t3], [t4, t5]])

    IOHyperEdge(value=[[t2, TBD], [TBD, t3]])

    edge1.match(edge2)

    assert sys.getrefcount(t1) != 2
    assert sys.getrefcount(t2) == 2
    assert sys.getrefcount(t3) == 2
    assert sys.getrefcount(t4) == 2
    assert sys.getrefcount(t5) == 2

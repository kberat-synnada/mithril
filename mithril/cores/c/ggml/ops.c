// Copyright 2022 Synnada, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ops.h"



struct ggml_tensor * add(struct ggml_context * ctx, struct ggml_tensor * left, struct ggml_tensor * right) {
    struct ggml_tensor * res = ggml_add(ctx, left, right);
    return res;
}


struct ggml_tensor * multiplication(struct ggml_context * ctx, struct ggml_tensor * left, struct ggml_tensor * right) {
    struct ggml_tensor * res = ggml_mul(ctx, left, right);
    return res;
}


struct ggml_tensor * broadcast_to(struct ggml_context * ctx, struct ggml_tensor * input, int dim1, int dim2, int dim3, int dim4) {
    // If the target shape is not 4D, the empty dimensions are need to be set to 1
    struct ggml_tensor * shape = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, 0, 0, 0, 0);
    shape->ne[0] = dim1;
    shape->ne[1] = dim2;
    shape->ne[2] = dim3;
    shape->ne[3] = dim4;
    struct ggml_tensor * res = ggml_repeat(ctx, input, shape);
    return res; 
}


struct ggml_tensor * add_grad(struct ggml_context * ctx, struct ggml_tensor * gradient, int idx, struct ggml_tensor * output, struct ggml_tensor * left, struct ggml_tensor * right)
{ 
    return gradient;
}


struct ggml_tensor * multiplication_grad(struct ggml_context * ctx, struct ggml_tensor * gradient, int idx, struct ggml_tensor * output, struct ggml_tensor * left, struct ggml_tensor * right)
{
    if (idx == 0){
        return multiplication(ctx, gradient, right);
    }
    else{
        return multiplication(ctx, gradient, left);
    }
}

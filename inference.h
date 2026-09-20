#pragma once

#include "model.h"

torch::Tensor prefill(
    Model& model,
    const torch::Tensor& input_ids,
    const torch::Tensor& padding_mask);

torch::Tensor decode(
    Model& model,
    const torch::Tensor& next_input_ids,
    const torch::Tensor& padding_mask);

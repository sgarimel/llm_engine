#include "inference.h"

torch::Tensor prefill(
    Model& model,
    const torch::Tensor& input_ids,
    const torch::Tensor& padding_mask) {
    return model.forward(input_ids, padding_mask, true);
}

torch::Tensor decode(
    Model& model,
    const torch::Tensor& next_input_ids,
    const torch::Tensor& padding_mask) {
    return model.forward(next_input_ids, padding_mask, true);
}
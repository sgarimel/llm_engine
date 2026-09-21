#pragma once

#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct TensorInfo {
    std::string dtype;
    std::vector<int64_t> shape;
    size_t start;
    size_t end;
};

torch::ScalarType parse_dtype(const std::string& dtype);

std::unordered_map<std::string, TensorInfo> read_json_tensors(
    std::string header_json);

torch::Tensor load_tensor(
    const std::string& name,
    const std::unordered_map<std::string, TensorInfo>& tensor_map,
    const char* data_ptr,
    size_t data_size);

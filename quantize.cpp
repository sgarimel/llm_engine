#include "safetensors.h"

#include <torch/torch.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

using namespace std;
using namespace nlohmann;

pair<torch::Tensor, torch::Tensor> quantize_weight(
    const torch::Tensor& weight) {
    if(weight.dim() != 2)
        throw runtime_error("Quantized weight must have two dimensions");

    torch::Tensor weight_float = weight.to(torch::kFloat32);
    torch::Tensor absmax = torch::amax(weight_float.abs(), 1);
    torch::Tensor scales = absmax / 127.0;
    scales = torch::where(
        absmax.eq(0),
        torch::ones_like(scales),
        scales
    );

    torch::Tensor quantized = torch::round(
        weight_float / scales.unsqueeze(1)
    ).clamp(-127, 127).to(torch::kInt8).contiguous();

    return {
        quantized,
        scales.to(torch::kBFloat16).contiguous()
    };
}

void save_safetensors(
    const string& output_path,
    const unordered_map<string, torch::Tensor>& tensors) {
    json header;
    header["__metadata__"] = {
        {"format", "pt"},
        {"quantization", "int8_weight_only"},
        {"scale_scheme", "per_channel_symmetric_row"},
        {"activation_dtype", "BF16"}
    };

    vector<string> names;
    names.reserve(tensors.size());
    for(const auto& [name, tensor] : tensors)
        names.push_back(name);
    sort(names.begin(), names.end());

    size_t offset = 0;
    for(const string& name : names) {
        const torch::Tensor& tensor = tensors.at(name);
        size_t bytes =
            static_cast<size_t>(tensor.numel()) *
            tensor.element_size();

        header[name] = {
            {"dtype", tensor.scalar_type() == torch::kInt8 ? "I8" : "BF16"},
            {"shape", tensor.sizes().vec()},
            {"data_offsets", vector<size_t>{offset, offset + bytes}}
        };
        offset += bytes;
    }

    string header_json = header.dump();
    while(header_json.size() % 8 != 0)
        header_json.push_back(' ');

    uint64_t header_length =
        static_cast<uint64_t>(header_json.size());
    ofstream output_file(output_path, ios::binary);
    if(!output_file)
        throw runtime_error("Could not create output: " + output_path);

    output_file.write(
        reinterpret_cast<const char*>(&header_length),
        sizeof(header_length)
    );
    output_file.write(header_json.data(), header_json.size());

    for(const string& name : names) {
        torch::Tensor tensor = tensors.at(name).contiguous();
        size_t bytes =
            static_cast<size_t>(tensor.numel()) *
            tensor.element_size();
        output_file.write(
            static_cast<const char*>(tensor.data_ptr()),
            static_cast<streamsize>(bytes)
        );
    }

    if(!output_file)
        throw runtime_error("Failed while writing: " + output_path);
}

int main(int argc, char** argv) {
    string input_path =
        "models/qwen2.5-1.5b-instruct/model.safetensors";
    string output_path =
        "models/qwen2.5-1.5b-instruct/model-int8.safetensors";

    if(argc > 1) input_path = argv[1];
    if(argc > 2) output_path = argv[2];

    int fd = open(input_path.c_str(), O_RDONLY);
    if(fd < 0)
        throw runtime_error("Could not open input: " + input_path);

    struct stat st{};
    if(fstat(fd, &st) != 0) {
        close(fd);
        throw runtime_error("Could not stat input: " + input_path);
    }

    size_t file_size = static_cast<size_t>(st.st_size);
    void* mapped_file = mmap(
        nullptr,
        file_size,
        PROT_READ,
        MAP_PRIVATE,
        fd,
        0
    );
    if(mapped_file == MAP_FAILED) {
        close(fd);
        throw runtime_error("Could not mmap input: " + input_path);
    }

    const char* file_ptr = static_cast<const char*>(mapped_file);
    uint64_t header_length;
    memcpy(&header_length, file_ptr, sizeof(header_length));

    string header_json(
        file_ptr + sizeof(header_length),
        header_length
    );
    const char* data_ptr =
        file_ptr + sizeof(header_length) + header_length;
    size_t data_size =
        file_size - sizeof(header_length) - header_length;

    unordered_map<string, TensorInfo> tensor_map =
        read_json_tensors(header_json);
    vector<string> names;
    names.reserve(tensor_map.size());
    for(const auto& [name, info] : tensor_map)
        names.push_back(name);

    unordered_map<string, torch::Tensor> outputs;
    outputs.reserve(tensor_map.size() * 2);

    for(const string& name : names) {
        outputs[name] =
            load_tensor(name, tensor_map, data_ptr, data_size);
    }

    const vector<string> projections = {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj"
    };

    int quantized_count = 0;
    for(int layer = 0; layer < 28; layer++) {
        string prefix = "model.layers." + to_string(layer) + ".";
        for(const string& projection : projections) {
            string weight_name = prefix + projection + ".weight";
            auto [quantized, scales] =
                quantize_weight(outputs.at(weight_name));
            outputs[weight_name] = quantized;
            outputs[prefix + projection + ".weight_scale"] = scales;
            quantized_count++;
            cout << "Quantized " << weight_name << endl;
        }
    }

    munmap(mapped_file, file_size);
    close(fd);

    save_safetensors(output_path, outputs);
    cout << "Saved " << quantized_count
         << " quantized weights to "
         << output_path << endl;
    return 0;
}

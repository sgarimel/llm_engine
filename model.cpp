#include "model.h"
#include <cstring>
#include <unordered_map>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept> 


using namespace std; 
using namespace nlohmann; 

torch::Tensor Model::forward(const torch::Tensor& x) { 
    throw std::logic_error("Model::forward not implemented");
}




torch::Tensor Block::forward(const torch::Tensor& x) { 
    throw std::logic_error("Model::forward not implemented");
}

torch::Tensor Attention::forward(const torch::Tensor& x) { 
    throw std::logic_error("Model::forward not implemented");
}

torch::Tensor MLP::forward(const torch::Tensor& x) { 
    throw std::logic_error("Model::forward not implemented");
}

torch::Tensor LayerNorm::forward(const torch::Tensor& x) {
    throw std::logic_error("Model::forward not implemented");
}


ModelConfig Loader::load_config() { 
    std::ifstream file(config_path);
    if (!file)
        throw std::runtime_error("Could not open config: " + config_path);

    json config_json;
    file >> config_json;

    ModelConfig config;
    config.vocab_size = config_json.at("vocab_size").get<int>();
    config.hidden_size = config_json.at("hidden_size").get<int>();
    config.intermediate_state = config_json.at("intermediate_size").get<int>();
    config.num_layers = config_json.at("num_hidden_layers").get<int>();
    config.num_query_heads = config_json.at("num_attention_heads").get<int>();
    config.num_kv_heads = config_json.at("num_key_value_heads").get<int>();
    config.head_dim = config.hidden_size / config.num_query_heads;
    config.max_seq_len = config_json.at("max_position_embeddings").get<int>();
    config.rms_norm_eps = config_json.at("rms_norm_eps").get<double>();
    config.rope_theta = config_json.at("rope_theta").get<double>();
    return config;
}

torch::ScalarType parse_dtype(const std::string& dtype)
{
    if (dtype == "BF16") return torch::kBFloat16;
    if (dtype == "F16")  return torch::kFloat16;
    if (dtype == "F32")  return torch::kFloat32;
    throw std::runtime_error("Unsupported dtype: " + dtype);
}

struct TensorInfo { 
    string dtype; 
    vector<int64_t> shape; 
    size_t start; 
    size_t end; 
};


unordered_map<string, TensorInfo> read_json_tensors(string header_json) { 
    json header = json::parse(header_json);
    unordered_map<string, TensorInfo> tensors; 

    for(const auto& [tensorName, tensorMetadata]: header.items()) { 
        if(tensorName == "__metadata__") continue; 

        auto offsets = tensorMetadata.at("data_offsets")
                               .get<vector<size_t>>();
        if(offsets.size() != 2 || offsets[0] > offsets[1]) { 
            cout << "invalid offsets for tensor with name " << tensorName << endl; 
            exit(0);
        }
        tensors.emplace(tensorName, TensorInfo{
            .dtype = tensorMetadata.at("dtype").get<std::string>(),
            .shape = tensorMetadata.at("shape").get<std::vector<int64_t>>(),
            .start = offsets[0],
            .end = offsets[1]
        });
    }
    return tensors; 
}

// Given a file pointer to tensors and a byte range, loads in the tensor 
torch::Tensor load_tensor(
    const string& name, 
    const std::unordered_map<std::string, TensorInfo>& tensor_map, 
    const char* data_ptr, 
    size_t data_size
) { 
    const TensorInfo& info = tensor_map.at(name);

    if(info.end > data_size)
        throw std::runtime_error("Tensor outside file: " + name);

    const char* tensor_data = data_ptr + info.start; 

    return torch::from_blob(
        const_cast<char*>(tensor_data), 
        info.shape, 
        torch::TensorOptions().dtype(parse_dtype(info.dtype))
    );
}

Model Loader::load_model() { 
    int fd = open(this->weights_path.c_str(), O_RDONLY);
    if (fd < 0) { 
        cout << "error in model loading weights" << endl;
        exit(0);
    }

    struct stat st{};
    fstat(fd, &st);
    size_t size = static_cast<size_t>(st.st_size);


    void* mapped_file = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);

    if(mapped_file == MAP_FAILED) { 
        cout << "couldn't mmap data into memory." << endl;
        exit(0);
    }

    // do safetensor parsing

    // 1. read N = u64 int containing the size of the header
    const char* file_ptr = static_cast<const char*>(mapped_file); 
    uint64_t header_len; 
    memcpy(&header_len, file_ptr, sizeof(uint64_t));


    // 2. for the next N bytes there will be "TENSOR_NAME": { "dtype": DATA_TYPE, "shape": List<Integer>, "data_offsets": [BEGIN, END]} basicallyt repeated 

    string header_json(
        file_ptr + sizeof(header_len),
        header_len
    );

    const char* data_ptr = file_ptr + sizeof(header_len) + header_len; 
    uint64_t data_size = size - sizeof(header_len) - header_len; 

    unordered_map<string, TensorInfo> tensors = read_json_tensors(header_json);
    ModelConfig config = load_config();
    std::vector<Block> blocks;
    blocks.reserve(config.num_layers);

    for(int layer = 0; layer < config.num_layers; layer++) { 
        string prefix = 
            "model.layers." + to_string(layer) + ".";
        
        torch::Tensor q_weight = load_tensor( 
            prefix + "self_attn.q_proj.weight", 
            tensors, data_ptr, data_size
        );

        torch::Tensor q_bias = load_tensor(
            prefix + "self_attn.q_proj.bias", 
            tensors, data_ptr, data_size
        );

        torch::Tensor k_weight = load_tensor(
            prefix + "self_attn.k_proj.weight",
            tensors, data_ptr, data_size
        );

        torch::Tensor k_bias = load_tensor(
            prefix + "self_attn.k_proj.bias",
            tensors, data_ptr, data_size
        );

        torch::Tensor v_weight = load_tensor(
            prefix + "self_attn.v_proj.weight",
            tensors, data_ptr, data_size
        );

        torch::Tensor v_bias = load_tensor(
            prefix + "self_attn.v_proj.bias",
            tensors, data_ptr, data_size
        );

        torch::Tensor o_weight = load_tensor(
            prefix + "self_attn.o_proj.weight",
            tensors, data_ptr, data_size
        );

        torch::Tensor down_proj = load_tensor(
            prefix + "mlp.down_proj.weight",
            tensors, data_ptr, data_size
        );

        torch::Tensor gate_proj = load_tensor(
            prefix + "mlp.gate_proj.weight",
            tensors, data_ptr, data_size
        );

        torch::Tensor up_proj = load_tensor(
            prefix + "mlp.up_proj.weight",
            tensors, data_ptr, data_size
        );

        LayerNorm pre_attention_norm(
            load_tensor(
                prefix + "input_layernorm.weight",
                tensors, data_ptr, data_size
            ),
            config.rms_norm_eps
        );

        LayerNorm post_attention_norm(
            load_tensor(
                prefix + "post_attention_layernorm.weight",
                tensors, data_ptr, data_size
            ),
            config.rms_norm_eps
        );

        Attention attention(
            q_weight,
            q_bias,
            k_weight,
            k_bias,
            v_weight,
            v_bias,
            o_weight
        );

        MLP mlp(
            gate_proj,
            up_proj,
            down_proj
        );


        blocks.emplace_back(
            attention,
            mlp,
            pre_attention_norm,
            post_attention_norm
        );
    }

    
    string embed_layer = "model.embed_tokens.weight";
    string final_norm_name = "model.norm.weight";

    torch::Tensor embeddings = load_tensor( 
        embed_layer, tensors, data_ptr, data_size
    );

    LayerNorm final_norm(
        load_tensor(
            final_norm_name,
            tensors,
            data_ptr,
            data_size
        ),
        config.rms_norm_eps
    );


    Model m(
        config, 
        embeddings, 
        blocks, 
        final_norm
    );
    
    munmap(mapped_file, size);
    close(fd);

    return m;
}
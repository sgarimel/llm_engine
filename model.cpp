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
#include <limits>
#include <cmath>
#include <unordered_set>


using namespace std; 
using namespace nlohmann; 

const unordered_map<string, function<torch::Tensor(const torch::Tensor&)> > activation_to_function = {
    {"silu", [](const torch::Tensor& x) { return torch::silu(x); }},
    {"relu", [](const torch::Tensor& x) { return torch::relu(x); }},
    {"gelu", [](const torch::Tensor& x) { return torch::gelu(x); }},
    {"tanh", [](const torch::Tensor& x) { return torch::tanh(x); }}
};

torch::Tensor Model::forward(const torch::Tensor& input_ids, const torch::Tensor& padding_mask) {


    torch::Tensor x = torch::nn::functional::embedding(
                        input_ids, 
                        this->embeddings
                      );

    auto mask_options = torch::TensorOptions().dtype(torch::kBool).device(input_ids.device());
    int seq_len = input_ids.size(1);
    int batch_size = input_ids.size(0);
    torch::Tensor causal_mask = torch::triu(
        torch::ones(
            {seq_len, seq_len}, 
            mask_options
        ), 
        1
    ).view({1, 1, seq_len, seq_len});

    // has dimension (B, N)
    torch::Tensor inverted_padding_mask = padding_mask.eq(0); // go from 0 = blocked, 1 = allowed to 0 = allowed, 1 = blocked
    inverted_padding_mask = inverted_padding_mask.view({
        input_ids.size(0),
        1,
        1,
        seq_len
    });

    torch::Tensor combined_mask = torch::logical_or(causal_mask, inverted_padding_mask);

    for(Block& block : this->blocks) { 
        x = block.forward(x, combined_mask);
    }

    x = this->norm.forward(x); 
    torch::Tensor logits = torch::matmul(x, this->embeddings.transpose(-2, -1));

    return logits.select(1, -1);
}

torch::Tensor Block::forward(const torch::Tensor& x, const torch::Tensor& attention_mask) {
    torch::Tensor pre_normed = this->pre_attention_norm.forward(x);
    torch::Tensor attention_output = this->attention.forward(pre_normed, attention_mask);
    torch::Tensor with_res_connection = attention_output + x; 
    torch::Tensor post_normed = this->post_attention_norm.forward(with_res_connection);
    torch::Tensor mlp_output = this->mlp.forward(post_normed);
    return with_res_connection + mlp_output;
}

torch::Tensor Attention::rope_embeddings(const torch::Tensor& x) { 
    // Pre compute frequencies theta_i
    // torch.arange(0, d_model, 2)
    int64_t head_dim = x.size(-1);
    int64_t num_tokens = x.size(-2); 
    // Convert to float for computing frequencies
    torch::Tensor dim_pair = torch::arange(0, head_dim, 2, torch::TensorOptions().dtype(torch::kFloat32).device(x.device())); // [0, 2, ...., head_dim - 2]
    float base = this->rope_theta;
    torch::Tensor frequencies = torch::pow(base, -dim_pair / static_cast<double>(head_dim)).unsqueeze(0);
    torch::Tensor token_position = torch::arange(0, num_tokens, torch::TensorOptions().dtype(torch::kFloat32).device(x.device())).unsqueeze(1);
    // Compute cos(m * freq), sin(m * freq) 
    torch::Tensor cosine_freq = torch::cos(torch::matmul(token_position, frequencies)).to(x.scalar_type());
    torch::Tensor sine_freq = torch::sin(torch::matmul(token_position, frequencies)).to(x.scalar_type());
    // Break up the dimensions
    
    // 1 - get the first half dimensions 
    torch::Tensor first_half = x.slice(-1, 0, head_dim / 2);
    torch::Tensor second_half = x.slice(-1, head_dim / 2, head_dim);

    // Compute hadamard onto each pair 
    torch::Tensor rotated_first_half = first_half * cosine_freq - second_half * sine_freq; 
    torch::Tensor rotated_second_half = first_half * sine_freq + second_half * cosine_freq; 

    torch::Tensor output = torch::cat({rotated_first_half, rotated_second_half}, -1); 
    return output;

}

torch::Tensor Attention::forward(
    const torch::Tensor& x,
    const torch::Tensor& attention_mask) { 
    // dimension of x is (B, N, d_model)
    // dimension of wq is (d_model, d_model), bq is (d_model)
    torch::Tensor q = torch::matmul(x, this->wq.transpose(-2, -1)) + this->bq;
    torch::Tensor k = torch::matmul(x, this->wk.transpose(-2, -1)) + this->bk;
    torch::Tensor v = torch::matmul(x, this->wv.transpose(-2, -1)) + this->bv;

    // we now want to split q k v into heads 
    q = q.view({q.size(0), q.size(1), this->num_query_heads, this->head_dim}).transpose(1, 2);
    k = k.view({k.size(0), k.size(1), this->num_kv_heads, this->head_dim}).transpose(1, 2);
    v = v.view({v.size(0), v.size(1), this->num_kv_heads, this->head_dim}).transpose(1, 2);

    q = rope_embeddings(q);
    k = rope_embeddings(k); 

    int repetitions = this->num_query_heads / this->num_kv_heads; // should be 6
    k = k.repeat_interleave(repetitions, 1);
    v = v.repeat_interleave(repetitions, 1);

    torch::Tensor scores = torch::einsum("bhid, bhjd -> bhij", {q, k});
    scores /= std::sqrt(static_cast<double>(this->head_dim));
    scores.masked_fill_(
        attention_mask,
        std::numeric_limits<c10::BFloat16>::lowest()
    );

    // take final row-wise softmax
    torch::Tensor softmax = torch::softmax(scores, -1, torch::kFloat32).to(q.scalar_type());
    torch::Tensor attention = torch::matmul(softmax, v);

    //repackage as proper view
    attention = attention.transpose(1, 2).contiguous();
    attention = attention.view({attention.size(0), attention.size(1), this->num_query_heads * this->head_dim}).contiguous();

    torch::Tensor linear_projection = torch::matmul(attention, this->wo.transpose(-2, -1));
    return linear_projection;
    
}

torch::Tensor MLP::forward(const torch::Tensor& x) { 
    // x = (B, N, d_model)
    // gate_proj = (intermediate_size, d_model)
    // up_proj = (intermediate_size, d_model)
    // down_proj = (d_model, intermediate_size)
    torch::Tensor swished = activation_to_function.at(this->hidden_act)(
        torch::matmul(x, this->gate_proj.transpose(-2, -1))
    ); // (B, N, intermediate_size)
    torch::Tensor up = torch::matmul(x, this->up_proj.transpose(-2, -1)); // (B, N, intermediate)
    torch::Tensor hadamard = swished * up; // (B, N, intermediate)
    return torch::matmul(hadamard, this->down_proj.transpose(-2, -1)); // (B, N, d_model)
    
}

torch::Tensor LayerNorm::forward(const torch::Tensor& x) {
    auto input_type = x.scalar_type();
    auto x_float = x.to(torch::kFloat32);
    auto var = torch::mean(torch::pow(x_float, 2), -1, true); // needs to be (B, N, 1) since we norm each token and need to broadcasted along final dimension
    auto normalized = x_float * torch::rsqrt(var + this->eps); 
    // this->weights has dimension (d_model)
    torch::Tensor output = normalized * weights; // (B, N, d_model) 
    return output.to(input_type);
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
    config.hidden_act = config_json.at("hidden_act").get<string>();
    config.rms_norm_eps = config_json.at("rms_norm_eps").get<double>();
    config.rope_theta = config_json.at("rope_theta").get<double>();
    return config;
}

GenerationConfig Loader::load_generation_config() {
    std::ifstream file(generation_config_path);
    if (!file)
        throw std::runtime_error(
            "Could not open generation config: " + generation_config_path
        );

    json config_json;
    file >> config_json;

    GenerationConfig config;
    config.bos_token_id = config_json.at("bos_token_id").get<int>();
    config.pad_token_id = config_json.at("pad_token_id").get<int>();
    config.do_sample = config_json.at("do_sample").get<bool>();
    config.repetition_penalty =
        config_json.at("repetition_penalty").get<double>();
    config.temperature = config_json.at("temperature").get<double>();
    config.top_p = config_json.at("top_p").get<double>();
    config.top_k = config_json.at("top_k").get<int>();

    const auto& eos = config_json.at("eos_token_id");
    if (eos.is_array())
        config.eos_token_ids = eos.get<vector<int>>();
    else
        config.eos_token_ids = {eos.get<int>()};

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
    ).clone();
}

void validate_tensor_names(
    const unordered_map<string, TensorInfo>& tensors,
    const ModelConfig& config)
{
    unordered_set<string> expected = {
        "model.embed_tokens.weight",
        "model.norm.weight"
    };
    const vector<string> layer_weights = {
        "self_attn.q_proj.weight",
        "self_attn.q_proj.bias",
        "self_attn.k_proj.weight",
        "self_attn.k_proj.bias",
        "self_attn.v_proj.weight",
        "self_attn.v_proj.bias",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight"
    };
    for (int layer = 0; layer < config.num_layers; ++layer) {
        string prefix =
            "model.layers." + to_string(layer) + ".";
        for (const string& suffix : layer_weights)
            expected.insert(prefix + suffix);
    }
    for (const string& name : expected) {
        if (!tensors.contains(name))
            throw runtime_error("Missing tensor: " + name);
    }
    for (const auto& [name, info] : tensors) {
        if (!expected.contains(name))
            throw runtime_error("Tensor was not expected: " + name);
    }
    cout << "Validated " << expected.size()
         << " tensor names\n";
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
    GenerationConfig generation_config = load_generation_config();
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
            o_weight,
            config.num_query_heads,
            config.num_kv_heads,
            config.head_dim,
            config.rope_theta
        );

        MLP mlp(
            gate_proj,
            up_proj,
            down_proj,
            config.hidden_act
        );


        blocks.emplace_back(
            attention,
            mlp,
            pre_attention_norm,
            post_attention_norm
        );
        cout << "Loaded layer number " << layer << endl; 
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
        generation_config,
        embeddings, 
        blocks, 
        final_norm
    );
    validate_tensor_names(tensors, config);
    
    munmap(mapped_file, size);
    close(fd);

    return m;
}
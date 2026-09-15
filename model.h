#include <torch/torch.h>
#include <iostream> 
#include <functional>
#include <unordered_map>

using namespace std; 


extern const unordered_map<string, ActivationFunction>
    activation_to_function;

struct ModelConfig { 
    int vocab_size; 
    int hidden_size; 
    int intermediate_state; 
    int num_layers; 
    int num_query_heads; 
    int num_kv_heads; 
    int head_dim; 
    int max_seq_len; 

    string hidden_act;
    double rms_norm_eps; 
    double rope_theta;
};

struct GenerationConfig {
    int bos_token_id;
    int pad_token_id;
    vector<int> eos_token_ids;
    bool do_sample;
    double repetition_penalty;
    double temperature;
    double top_p;
    int top_k;
};

class LayerNorm {
    public:
        LayerNorm(torch::Tensor weights, double eps)
            : weights(weights), eps(eps) {}

        torch::Tensor forward(const torch::Tensor& x);

    private:
        torch::Tensor weights;
        double eps;
};

class Attention { 
    public:
        Attention(
            torch::Tensor wq,
            torch::Tensor bq,
            torch::Tensor wk,
            torch::Tensor bk,
            torch::Tensor wv,
            torch::Tensor bv,
            torch::Tensor wo, 
            int num_query_heads, 
            int num_kv_heads, 
            int head_dim, 
            double rope_theta)
            : wq(wq),
            bq(bq),
            wk(wk),
            bk(bk),
            wv(wv),
            bv(bv),
            wo(wo),
            num_query_heads(num_query_heads),
            num_kv_heads(num_kv_heads),
            head_dim(head_dim),
            rope_theta(rope_theta) {}
 
        torch::Tensor forward(
            const torch::Tensor& x,
            const torch::Tensor& attention_mask);
    private: 
        torch::Tensor rope_embeddings(const torch::Tensor& x);

        torch::Tensor wq; 
        torch::Tensor bq; 
        torch::Tensor wk; 
        torch::Tensor bk;
        torch::Tensor wv;
        torch::Tensor bv; 
        torch::Tensor wo; 

        int num_query_heads;
        int num_kv_heads;
        int head_dim;
        double rope_theta;
};

class MLP {
    public: 
        MLP(
            torch::Tensor gate,
            torch::Tensor up,
            torch::Tensor down,
            string hidden_act)
            : gate_proj(gate),
            up_proj(up),
            down_proj(down),
            hidden_act(hidden_act) {}

        torch::Tensor forward(const torch::Tensor& x);
    private: 
        torch::Tensor gate_proj; 
        torch::Tensor up_proj;
        torch::Tensor down_proj; 
        string hidden_act;
};

class Block {
    public:
        Block(
            Attention attention,
            MLP mlp,
            LayerNorm input_norm,
            LayerNorm post_norm)
            : attention(attention),
              mlp(mlp),
              pre_attention_norm(input_norm),
              post_attention_norm(post_norm) {}

        torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& attention_mask);

    private:
        Attention attention;
        MLP mlp;
        LayerNorm pre_attention_norm;
        LayerNorm post_attention_norm;
};

class Model {
    public:
        Model(
            ModelConfig config,
            GenerationConfig generation_config,
            torch::Tensor embeddings,
            std::vector<Block> blocks,
            LayerNorm norm)
            : config(config),
              generation_config(generation_config),
              embeddings(embeddings),
              blocks(blocks),
              norm(norm) {}

        torch::Tensor forward(const torch::Tensor& x);

    private:
        ModelConfig config;
        GenerationConfig generation_config;
        torch::Tensor embeddings;
        std::vector<Block> blocks;
        LayerNorm norm;
};

class Loader {
    public:
        Loader(
            string config_path,
            string generation_config_path,
            string weights_path)
            : config_path(config_path),
              generation_config_path(generation_config_path),
              weights_path(weights_path) {}

        ModelConfig load_config();
        GenerationConfig load_generation_config();
        Model load_model();

    private:
        string config_path;
        string generation_config_path;
        string weights_path;
};
#include <torch/torch.h>
#include <iostream> 

using namespace std; 

struct ModelConfig { 
    int vocab_size; 
    int hidden_size; 
    int intermediate_state; 
    int num_layers; 
    int num_query_heads; 
    int num_kv_heads; 
    int head_dim; 
    int max_seq_len; 

    double rms_norm_eps; 
    double rope_theta;
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
            torch::Tensor wo)
            : wq(wq),
            bq(bq),
            wk(wk),
            bk(bk),
            wv(wv),
            bv(bv),
            wo(wo) {}
 
        torch::Tensor forward(const torch::Tensor& x);
    private: 
        torch::Tensor wq; 
        torch::Tensor bq; 
        torch::Tensor wk; 
        torch::Tensor bk;
        torch::Tensor wv;
        torch::Tensor bv; 
        torch::Tensor wo; 
};

class MLP {
    public: 
        MLP(
            torch::Tensor gate,
            torch::Tensor up,
            torch::Tensor down)
            : gate_proj(gate),
            up_proj(up),
            down_proj(down) {}

        torch::Tensor forward(const torch::Tensor& x);
    private: 
        torch::Tensor gate_proj; 
        torch::Tensor up_proj;
        torch::Tensor down_proj; 
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

        torch::Tensor forward(const torch::Tensor& x);

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
            torch::Tensor embeddings,
            std::vector<Block> blocks,
            LayerNorm norm)
            : config(config),
              embeddings(embeddings),
              blocks(blocks),
              norm(norm) {}

        torch::Tensor forward(const torch::Tensor& x);

    private:
        ModelConfig config;
        torch::Tensor embeddings;
        std::vector<Block> blocks;
        LayerNorm norm;
};

class Loader {
    public:
        Loader(string config_path, string weights_path)
            : config_path(config_path),
              weights_path(weights_path) {}

        ModelConfig load_config();
        Model load_model();

    private:
        string config_path;
        string weights_path;
};
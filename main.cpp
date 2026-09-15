#include "model.h"
#include "tokenizer.h"
#include <torch/torch.h>
#include <algorithm>
using namespace std; 

torch::Tensor sample_top_k(
    const torch::Tensor& logits, 
    int top_k, 
    double temperature
) { 
    torch::Tensor scaled_logits = logits / temperature; 
    auto [top_values, top_indices] = torch::topk(scaled_logits, top_k, -1); // [B, K]
    torch::Tensor probabilities = torch::softmax(top_values, -1, torch::kFloat32); // [B, K]
    torch::Tensor sampled_positions = torch::multinomial(probabilities, 1); // [B, 1]
    return top_indices.gather(1, sampled_positions).squeeze(1); // [B]
}

bool is_eos(int64_t token_id, const vector<int>& eos_token_ids) { 
    return find(eos_token_ids.begin(), eos_token_ids.end(), token_id) != eos_token_ids.end();
}

int main() { 
    string config_path = "models/qwen2.5-1.5b-instruct/config.json";
    string generation_config_path =
        "models/qwen2.5-1.5b-instruct/generation_config.json";
    string weights_path = "models/qwen2.5-1.5b-instruct/model.safetensors";
    string tokenizer_path = "models/qwen2.5-1.5b-instruct/tokenizer.json";
    Loader l(config_path, generation_config_path, weights_path);
    GenerationConfig generation_config = l.load_generation_config();
    Model m = l.load_model();
    cout << "Model loaded" << endl;

    QwenTokenizer tokenizer(tokenizer_path, generation_config.pad_token_id);
    vector<string> prompts = {"Who is the president of the united states?"};
    TokenizedInput tokens = tokenizer.tokenize(prompts);
    torch::Tensor input_ids = tokens.input_ids; 
    torch::Tensor padding_mask = tokens.padding_mask;

    vector<bool> finished(prompts.size(), false);
    int max_new_tokens = 100;

    for(int step = 0; step < max_new_tokens; step++) { 
        cout << "Generation step " << step + 1
             << "/" << max_new_tokens << endl;
        torch::Tensor logits = m.forward(input_ids, padding_mask);
        torch::Tensor next_ids;

        if (generation_config.do_sample) {
            next_ids = sample_top_k(
                logits,
                generation_config.top_k,
                generation_config.temperature
            ); // [B]
        } else {
            next_ids = logits.argmax(-1);
        }

        torch::Tensor new_mask = torch::ones_like(next_ids); 

        for (int64_t batch = 0; batch < next_ids.size(0); ++batch) {
            if (finished[batch]) {
                next_ids[batch] = generation_config.pad_token_id;
                new_mask[batch] = 0;
                continue;
            }

            int64_t next_id = next_ids[batch].item<int64_t>();
            if (is_eos(next_id, generation_config.eos_token_ids))
                finished[batch] = true;
        }

        input_ids = torch::cat({input_ids, next_ids.unsqueeze(1)}, 1);
        padding_mask = torch::cat({padding_mask, new_mask.unsqueeze(1)}, 1); //[B,N] to [B,N+1]

        for (int64_t batch = 0; batch < input_ids.size(0); ++batch) {
            vector<int64_t> sequence;
            for (int64_t token = 0; token < input_ids.size(1); ++token) {
                if (padding_mask[batch][token].item<int64_t>() == 1) {
                    sequence.push_back(
                        input_ids[batch][token].item<int64_t>()
                    );
                }
            }

            cout << "Batch " << batch << ": "
                 << tokenizer.decode(sequence) << endl;
        }

        bool all_finished = true;
        for (bool sequence_finished : finished)
            all_finished = all_finished && sequence_finished;

        if (all_finished)
            break;
    }

    for (int64_t batch = 0; batch < input_ids.size(0); ++batch) {
        vector<int64_t> sequence;
        for (int64_t token = 0; token < input_ids.size(1); ++token) {
            if (padding_mask[batch][token].item<int64_t>() == 1) {
                sequence.push_back(
                    input_ids[batch][token].item<int64_t>()
                );
            }
        }

        if (!sequence.empty() &&
            is_eos(sequence.back(), generation_config.eos_token_ids)) {
            sequence.pop_back();
        }

        cout << "\nOutput " << batch << ":\n"
             << tokenizer.decode(sequence) << endl;
    }

    return 0;
}
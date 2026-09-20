#include "model.h"
#include "tokenizer.h"
#include "inference.h"
#include <torch/torch.h>
#include <algorithm>
#include <chrono>
using namespace std; 

struct Benchmark {
    int max_new_tokens;
    vector<int64_t> prefill_tokens;
    vector<vector<float>> decode_token_times_seconds;
};

struct TrialMetrics {
    float tokenization_seconds;
    float cache_initialization_seconds;
    float prefill_forward_seconds;
    float decode_forward_seconds;
    float total_generation_seconds;
    float ttgt_seconds;
    float prefill_throughput;
    float tpot_seconds;
    float decode_throughput;
    int64_t decode_tokens;
};

struct BenchmarkResults {
    vector<float> tokenization_seconds;
    vector<float> cache_initialization_seconds;
    vector<float> prefill_forward_seconds;
    vector<float> decode_forward_seconds;
    vector<float> total_generation_seconds;
    vector<float> ttgt_seconds;
    vector<float> prefill_throughput;
    vector<float> tpot_seconds;
    vector<float> decode_throughput;
    vector<int64_t> decode_tokens;
};
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

TrialMetrics run_benchmark(
    Model& m,
    QwenTokenizer& tokenizer,
    const GenerationConfig& generation_config,
    const vector<string>& prompts,
    bool use_cache) {
    TrialMetrics metrics{};
    auto total_start = chrono::steady_clock::now();

    auto tokenization_start = chrono::steady_clock::now();
    TokenizedInput tokens = tokenizer.tokenize(prompts);
    auto tokenization_end = chrono::steady_clock::now();
    metrics.tokenization_seconds =
        chrono::duration<float>(
            tokenization_end - tokenization_start
        ).count();

    torch::Tensor input_ids = tokens.input_ids; 
    torch::Tensor padding_mask = tokens.padding_mask;
    vector<bool> finished(input_ids.size(0), false);

    Benchmark benchmark;
    benchmark.max_new_tokens = 150;
    benchmark.decode_token_times_seconds.resize(input_ids.size(0));

    torch::Tensor prefill_tokens = padding_mask.sum(1);
    for (int64_t batch = 0; batch < prefill_tokens.size(0); ++batch)
        benchmark.prefill_tokens.push_back(prefill_tokens[batch].item<int64_t>());

    if(use_cache) {
        auto cache_start = chrono::steady_clock::now();
        m.initialize_cache(input_ids.size(0));
        auto cache_end = chrono::steady_clock::now();
        metrics.cache_initialization_seconds =
            chrono::duration<float>(cache_end - cache_start).count();
    }

    auto prefill_start = chrono::steady_clock::now();
    torch::Tensor logits;
    if(use_cache) logits = prefill(m, input_ids, padding_mask);
    else logits = m.forward(input_ids, padding_mask, false);
    auto prefill_end = chrono::steady_clock::now();

    metrics.prefill_forward_seconds =
        chrono::duration<float>(prefill_end - prefill_start).count();
    metrics.ttgt_seconds =
        chrono::duration<float>(prefill_end - total_start).count();

    for(int step = 0; step < benchmark.max_new_tokens; step++) { 
        torch::Tensor next_ids;
        if(generation_config.do_sample) {
            next_ids = sample_top_k(
                logits,
                generation_config.top_k,
                generation_config.temperature
            );
        } else {
            next_ids = logits.argmax(-1);
        }

        torch::Tensor new_mask = torch::ones_like(next_ids); 
        for(int64_t batch = 0; batch < next_ids.size(0); batch++) {
            if(finished[batch]) {
                next_ids[batch] = generation_config.pad_token_id;
                new_mask[batch] = 0;
                continue;
            }

            int64_t next_id = next_ids[batch].item<int64_t>();
            if(is_eos(next_id, generation_config.eos_token_ids))
                finished[batch] = true;
        }

        torch::Tensor next_input_ids = next_ids.unsqueeze(1);
        input_ids = torch::cat({input_ids, next_input_ids}, 1);
        padding_mask = torch::cat({padding_mask, new_mask.unsqueeze(1)}, 1);

        bool all_finished = true;
        for(bool sequence_finished : finished)
            all_finished = all_finished && sequence_finished;

        if(all_finished || step + 1 == benchmark.max_new_tokens)
            break;

        auto decode_start = chrono::steady_clock::now();
        if(use_cache) logits = decode(m, next_input_ids, padding_mask);
        else logits = m.forward(input_ids, padding_mask, false);
        auto decode_end = chrono::steady_clock::now();

        metrics.decode_forward_seconds +=
            chrono::duration<float>(decode_end - decode_start).count();
        float token_time =
            chrono::duration<float>(decode_end - total_start).count();

        for(int64_t batch = 0; batch < logits.size(0); batch++) {
            if(!finished[batch])
                benchmark.decode_token_times_seconds[batch].push_back(token_time);
        }
    }

    metrics.total_generation_seconds =
        chrono::duration<float>(
            chrono::steady_clock::now() - total_start
        ).count();

    int64_t total_prefill_tokens = 0;
    for(int64_t token_count : benchmark.prefill_tokens)
        total_prefill_tokens += token_count;
    metrics.prefill_throughput =
        total_prefill_tokens / metrics.prefill_forward_seconds;

    float total_itl = 0.0f;
    float decode_end = metrics.ttgt_seconds;
    for(const vector<float>& token_times : benchmark.decode_token_times_seconds) {
        float previous_time = metrics.ttgt_seconds;
        for(float token_time : token_times) {
            total_itl += token_time - previous_time;
            previous_time = token_time;
        }

        metrics.decode_tokens += token_times.size();
        if(!token_times.empty())
            decode_end = max(decode_end, token_times.back());
    }

    float decode_duration = decode_end - metrics.ttgt_seconds;
    metrics.tpot_seconds = metrics.decode_tokens == 0
        ? 0.0f
        : total_itl / metrics.decode_tokens;
    metrics.decode_throughput = decode_duration == 0.0f
        ? 0.0f
        : metrics.decode_tokens / decode_duration;

    return metrics;
}

void store_result(BenchmarkResults& results, const TrialMetrics& metrics) {
    results.tokenization_seconds.push_back(metrics.tokenization_seconds);
    results.cache_initialization_seconds.push_back(metrics.cache_initialization_seconds);
    results.prefill_forward_seconds.push_back(metrics.prefill_forward_seconds);
    results.decode_forward_seconds.push_back(metrics.decode_forward_seconds);
    results.total_generation_seconds.push_back(metrics.total_generation_seconds);
    results.ttgt_seconds.push_back(metrics.ttgt_seconds);
    results.prefill_throughput.push_back(metrics.prefill_throughput);
    results.tpot_seconds.push_back(metrics.tpot_seconds);
    results.decode_throughput.push_back(metrics.decode_throughput);
    results.decode_tokens.push_back(metrics.decode_tokens);
}

float average(const vector<float>& values) {
    float total = 0.0f;
    for(float value : values)
        total += value;
    return total / values.size();
}

float average(const vector<int64_t>& values) {
    int64_t total = 0;
    for(int64_t value : values)
        total += value;
    return static_cast<float>(total) / values.size();
}

void print_results(const string& name, const BenchmarkResults& results) {
    cout << "\n" << name << " averages across "
         << results.ttgt_seconds.size() << " trials\n";
    cout << "Tokenization: "
         << average(results.tokenization_seconds) << " seconds\n";
    cout << "Cache initialization: "
         << average(results.cache_initialization_seconds) << " seconds\n";
    cout << "Prefill forward: "
         << average(results.prefill_forward_seconds) << " seconds\n";
    cout << "Decode forwards total: "
         << average(results.decode_forward_seconds) << " seconds\n";
    cout << "Total generation: "
         << average(results.total_generation_seconds) << " seconds\n";
    cout << "TTGT: "
         << average(results.ttgt_seconds) << " seconds\n";
    cout << "Prefill throughput: "
         << average(results.prefill_throughput) << " tokens/second\n";
    cout << "TPOT: "
         << average(results.tpot_seconds) << " seconds/token\n";
    cout << "Decode throughput: "
         << average(results.decode_throughput) << " tokens/second\n";
    cout << "Decode tokens: "
         << average(results.decode_tokens) << "\n";
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
    vector<string> prompts = {"How does transformer decoding work from start to end?."};

    int warmup_trials = 2;
    int measured_trials = 10;

    for(int trial = 0; trial < warmup_trials; trial++) {
        cout << "Warmup trial " << trial + 1 << "/" << warmup_trials << endl;
        torch::manual_seed(trial);
        run_benchmark(m, tokenizer, generation_config, prompts, false);
        torch::manual_seed(trial);
        run_benchmark(m, tokenizer, generation_config, prompts, true);
    }

    BenchmarkResults no_cache_results;
    BenchmarkResults kv_cache_results;

    for(int trial = 0; trial < measured_trials; trial++) {
        cout << "Measured trial " << trial + 1 << "/" << measured_trials << endl;

        torch::manual_seed(warmup_trials + trial);
        TrialMetrics no_cache = run_benchmark(
            m,
            tokenizer,
            generation_config,
            prompts,
            false
        );
        store_result(no_cache_results, no_cache);

        torch::manual_seed(warmup_trials + trial);
        TrialMetrics kv_cache = run_benchmark(
            m,
            tokenizer,
            generation_config,
            prompts,
            true
        );
        store_result(kv_cache_results, kv_cache);
    }

    print_results("No cache", no_cache_results);
    print_results("KV cache", kv_cache_results);

    return 0;
}
#pragma once

#include <torch/torch.h>
#include <tokenizers_cpp.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct TokenizedInput {
    torch::Tensor input_ids;
    torch::Tensor padding_mask;
};

class QwenTokenizer {
public:
    QwenTokenizer(
        const std::string& tokenizer_path,
        int64_t pad_token_id);

    TokenizedInput tokenize(const std::string& input);
    TokenizedInput tokenize(const std::vector<std::string>& inputs);
    std::string decode(const std::vector<int64_t>& token_ids);

private:
    std::unique_ptr<tokenizers::Tokenizer> tokenizer;
    int64_t pad_token_id;
};

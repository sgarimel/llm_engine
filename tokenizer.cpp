#include "tokenizer.h"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <stdexcept>

std::string read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("Could not open tokenizer: " + path);

    return std::string(
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>()
    );
}

QwenTokenizer::QwenTokenizer(
    const std::string& tokenizer_path,
    int64_t pad_token_id)
    : pad_token_id(pad_token_id) {
    tokenizer = tokenizers::Tokenizer::FromBlobJSON(
        read_file(tokenizer_path)
    );
}

TokenizedInput QwenTokenizer::tokenize(const std::string& input) {
    return tokenize(std::vector<std::string>{input});
}

TokenizedInput QwenTokenizer::tokenize(
    const std::vector<std::string>& inputs) {
    if (inputs.empty())
        throw std::runtime_error("Cannot tokenize an empty batch");

    auto encoded = tokenizer->EncodeBatch(inputs);

    int64_t batch_size = static_cast<int64_t>(encoded.size());
    int64_t max_length = 0;

    for (const auto& token_ids : encoded) {
        max_length = std::max(
            max_length,
            static_cast<int64_t>(token_ids.size())
        );
    }

    auto padded_ids = torch::full(
        {batch_size, max_length},
        pad_token_id,
        torch::kLong
    );
    auto padding_mask = torch::zeros(
        {batch_size, max_length},
        torch::kLong
    );

    for (int64_t batch = 0; batch < batch_size; ++batch) {
        int64_t length =
            static_cast<int64_t>(encoded[batch].size());
        int64_t left_padding = max_length - length;

        for (int64_t token = 0; token < length; ++token) {
            int64_t position = left_padding + token;

            padded_ids[batch][position] = encoded[batch][token];
            padding_mask[batch][position] = 1;
        }
    }

    return TokenizedInput{padded_ids, padding_mask};
}

std::string QwenTokenizer::decode(
    const std::vector<int64_t>& token_ids) {
    std::vector<int32_t> ids;
    ids.reserve(token_ids.size());

    for (int64_t token_id : token_ids)
        ids.push_back(static_cast<int32_t>(token_id));

    return tokenizer->Decode(ids);
}

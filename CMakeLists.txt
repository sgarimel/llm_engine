cmake_minimum_required(VERSION 3.19)
project(llmengine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

find_package(Torch REQUIRED)
find_package(nlohmann_json REQUIRED)

include(FetchContent)
FetchContent_Declare(
    tokenizers_cpp_dependency
    GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp.git
    GIT_TAG c586c52f93f7b060753bd2388eb96a105cb7374d
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(tokenizers_cpp_dependency)

add_library(llmengine model.cpp tokenizer.cpp inference.cpp)
add_library(
    llmengine_int8
    quantized_model.cpp
    tokenizer.cpp
    inference.cpp
)

target_link_libraries(
    llmengine
    PUBLIC
    "${TORCH_LIBRARIES}"
    tokenizers_cpp
)

target_link_libraries(
    llmengine_int8
    PUBLIC
    "${TORCH_LIBRARIES}"
    tokenizers_cpp
)

target_link_libraries(
    llmengine
    PRIVATE
    nlohmann_json::nlohmann_json
)

target_link_libraries(
    llmengine_int8
    PRIVATE
    nlohmann_json::nlohmann_json
)

add_executable(llmengine_app main.cpp)
add_executable(llmengine_int8_app main.cpp)
add_executable(quantize_int8 quantize.cpp)

target_link_libraries(
    llmengine_app
    PRIVATE
    llmengine
    "${TORCH_LIBRARIES}"
)

target_compile_definitions(llmengine_int8_app PRIVATE LLMENGINE_INT8=1)
target_link_libraries(
    llmengine_int8_app
    PRIVATE
    llmengine_int8
    "${TORCH_LIBRARIES}"
)

target_link_libraries(
    quantize_int8
    PRIVATE
    llmengine
    "${TORCH_LIBRARIES}"
    nlohmann_json::nlohmann_json
)

set_property(
    TARGET llmengine
    PROPERTY CXX_STANDARD 20
)

set_property(
    TARGET llmengine_int8
    PROPERTY CXX_STANDARD 20
)
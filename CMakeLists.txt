cmake_minimum_required(VERSION 3.18)
project(llmengine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)
find_package(nlohmann_json REQUIRED)

add_library(llmengine model.cpp)

target_link_libraries(
    llmengine
    PRIVATE
    "${TORCH_LIBRARIES}"
    nlohmann_json::nlohmann_json
)

add_executable(llmengine_app main.cpp)

target_link_libraries(
    llmengine_app
    PRIVATE
    llmengine
    "${TORCH_LIBRARIES}"
)

set_property(
    TARGET llmengine
    PROPERTY CXX_STANDARD 20
)
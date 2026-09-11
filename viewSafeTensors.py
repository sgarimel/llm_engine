from safetensors import safe_open
import torch

with safe_open("models/qwen2.5-1.5b-instruct/model.safetensors", framework="pt") as f:
    for name in f.keys():
        tensor = f.get_tensor(name)
        print(name, tensor.shape, tensor.dtype)
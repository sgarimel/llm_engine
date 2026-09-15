#include "model.h"

using namespace std; 

int main() { 
    string config_path = "models/qwen2.5-1.5b-instruct/config.json";
    string generation_config_path =
        "models/qwen2.5-1.5b-instruct/generation_config.json";
    string weights_path = "models/qwen2.5-1.5b-instruct/model.safetensors";
    Loader l(config_path, generation_config_path, weights_path);
    Model m = l.load_model();
    cout << "Model loaded" << endl;
}
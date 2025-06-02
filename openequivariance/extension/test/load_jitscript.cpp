#include <torch/script.h>

#include <iostream>
#include <memory>

/* 
* This program takes in two JITScript modules that execute 
* a single tensor product in FP32 precision. 
* The first module is compiled with OpenEquivariance, the second is
* e3nn's compiled module. The program will check that the
* two outputs are equal. 
*/

int main(int argc, const char* argv[]) {
    if (argc != 7) {
        std::cerr << "usage: load_jitscript "
                    << "<path-to-e3nn-module> "
                    << "<path-to-oeq-module> "
                    << "<L1_dim> "
                    << "<L2_dim> "
                    << "<weight_numel> "
                    << "<batch_size> "
                    << std::endl;

        return 1;
    }

    torch::jit::script::Module module;
    try {
        module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading script module" << std::endl;
        return 1;
    }

    int64_t L1_dim = std::stoi(argv[3]);
    int64_t L2_dim = std::stoi(argv[4]);
    int64_t weight_numel = std::stoi(argv[5]);
    int64_t batch_size = std::stoi(argv[6]); 

    torch::Device device(torch::kCUDA);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::randn({batch_size, L1_dim}, device));
    inputs.push_back(torch::randn({batch_size, L2_dim}, device));
    inputs.push_back(torch::randn({batch_size, weight_numel}, device));
    module.to(device);

    at::Tensor output = module.forward(inputs).toTensor();

    return 0;
}
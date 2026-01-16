#include <torch/script.h>

#include <iostream>
#include <memory>

/* 
* This program takes in two JITScript modules that execute 
* a tensor product in FP32 precision. 
* The first module is compiled from e3nn, the second is
* OEQ's compiled module. The program checks that the
* two outputs are comparable. 
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

    int64_t L1_dim = std::stoi(argv[3]);
    int64_t L2_dim = std::stoi(argv[4]);
    int64_t weight_numel = std::stoi(argv[5]);
    int64_t batch_size = std::stoi(argv[6]); 

    torch::Device device(torch::kCUDA);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::randn({batch_size, L1_dim}, device));
    inputs.push_back(torch::randn({batch_size, L2_dim}, device));
    inputs.push_back(torch::randn({batch_size, weight_numel}, device));

    torch::jit::script::Module module_e3nn, module_oeq;
    try {
        module_e3nn = torch::jit::load(argv[1]);
        module_oeq = torch::jit::load(argv[2]);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading script module" << std::endl;
        return 1;
    }

    module_e3nn.to(device);
    module_oeq.to(device);

    at::Tensor output_e3nn = module_e3nn.forward(inputs).toTensor();
    at::Tensor output_oeq = module_oeq.forward(inputs).toTensor();

    if(at::allclose(output_e3nn, output_oeq, 1e-5, 1e-5)) {
        return 0;
    } 
    else {
        std::cerr << "torch.allclose returned FALSE comparing model outputs." << std::endl;
        return 1;
    }
}
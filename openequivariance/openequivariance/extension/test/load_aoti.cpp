#include <vector>
#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

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
        std::cerr << "usage: load_aoti "
                    << "<path-to-e3nn-module> "
                    << "<path-to-oeq-module> "
                    << "<L1_dim> "
                    << "<L2_dim> "
                    << "<weight_numel> "
                    << "<batch_size> "
                    << std::endl;

        return 1;
    }

    c10::InferenceMode mode;

    int64_t L1_dim = std::stoi(argv[3]);
    int64_t L2_dim = std::stoi(argv[4]);
    int64_t weight_numel = std::stoi(argv[5]);
    int64_t batch_size = std::stoi(argv[6]); 

    std::vector<torch::Tensor> inputs;
    inputs.push_back(torch::randn({batch_size, L1_dim}, at::kCUDA));
    inputs.push_back(torch::randn({batch_size, L2_dim}, at::kCUDA));
    inputs.push_back(torch::randn({batch_size, weight_numel}, at::kCUDA));

    try { 
        torch::inductor::AOTIModelPackageLoader module_e3nn(argv[1]);
        torch::inductor::AOTIModelPackageLoader module_oeq(argv[2]);

        std::vector<torch::Tensor> output_e3nn = module_e3nn.run(inputs);
        std::vector<torch::Tensor> output_oeq = module_oeq.run(inputs); 

        for (size_t i = 0; i < output_e3nn.size(); i++) {
            if(at::allclose(output_e3nn[i], output_oeq[i], 1e-5, 1e-5)) {
                return 0;
            } 
            else {
                std::cerr << "torch.allclose returned FALSE comparing model outputs." << std::endl;
                return 1;
            }
        }
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading script module" << std::endl;
        return 1;
    }
}
#include <torch/script.h>

#include <iostream>
#include <memory>

/* 
* This program takes in two JITScript modules that execute 
* a single instruction, (5, 3, 5)-UVU tensor product in FP32 precision. 
* The first module is compiled with OpenEquivariance, the second is
* e3nn's compiled module. The program will check that the
* two outputs are equal. 
*/

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: example-app <path-to-exported-script-module>" << std::endl;
        return -1;
    }

    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    std::cout << "ok\n";
    return 0;
}
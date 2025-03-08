from openequivariance.implementations import LoopUnrollTP

impimentations = [
    LoopUnrollTP
]

def correctness(params):
    implementations = [LoopUnrollTP]
    directions = [ 'forward', 'backward']
    problems = [CTPP(*config) for config in mace_conv + nequip_conv]

    tests = [TestDefinition(implementation, problem, direction, correctness=True, benchmark=False) 
             for implementation, problem, direction
             in itertools.product(implementations, problems, directions)]

    bench_suite = TensorProductSuite(
        correctness_threshold = 5e-5,
        num_warmup=100,
        num_iter=100,
        prng_seed=11111,
        torch_op=False)

    bench_suite.run(tests, params.output_folder)
            
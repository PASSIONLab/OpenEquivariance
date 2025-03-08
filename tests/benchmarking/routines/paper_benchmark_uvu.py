def benchmark_uvu(params):
    implementations = [
        implementation_map[impl] for impl in params.implementations
    ] 
    directions = params.directions
    datatypes = [datatype_map[dt] for dt in params.datatypes]

    problems = []
    for dtype in datatypes:
        for config in mace_conv + nequip_conv:
            problem = CTPP(*config) # float32 by default

            if dtype == np.float64:
                problem.irrep_dtype = np.float64
                problem.weight_dtype = np.float64

            problems.append(problem)

    tests = [TestDefinition(implementation, problem, direction, correctness=False, benchmark=True) 
             for implementation, problem, direction
             in itertools.product(implementations, problems, directions)]

    # Handle the float64 Benzene case, since we run out of memory with torch compile
    tests = [test for test in tests
            if 'benzene' not in test.problem.label
            or test.implementation != E3NNTensorProductCompiledMaxAutotuneCUDAGraphs 
            or test.problem.irrep_dtype != np.float64]

    if 'e3nn' in params.implementations and 'float64' in params.datatypes:
        tests.extend([TestDefinition(E3NNTensorProduct, 
            CTPP('64x0o + 64x0e + 64x1o + 64x1e + 64x2o + 64x2e + 64x3o + 64x3e',  '0e + 1o + 2e + 3o', '64x0o + 64x0e + 64x1o + 64x1e + 64x2o + 64x2e + 64x3o + 64x3e', 
                    'nequip-revmd17-benzene', irrep_dtype=np.float64, weight_dtype=np.float64), direction, correctness=False, benchmark=True) 
                    for direction in ['forward', 'backward']])

    # Remove some more configurations for GPUs with limited memory 
    if params.limited_memory:
        tests = [test for test in tests if 
                (test.implementation == LoopUnrollTP and 'benzene' not in test.problem.label)
                or (test.implementation == CUETensorProduct and 'benzene' not in test.problem.label)
                or ('benzene' not in test.problem.label and test.problem.irrep_dtype != np.float64)]

    bench_suite = TensorProductSuite(
        correctness_threshold = 5e-5,
        num_warmup=100,
        num_iter=100,
        bench_batch_size=params.batch_size,
        prng_seed=11111,
        test_name="uvu")

    data_folder = bench_suite.run(tests, params.output_folder)

    if params.plot:
        plot({"data_folder": data_folder})
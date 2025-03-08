def benchmark_roofline(params):
    implementations =   [LoopUnrollTP, CUETensorProduct]
    directions = ['forward', 'backward']

    tests = [TestDefinition(implementation, problem, direction, correctness=False, benchmark=True) 
             for implementation, problem, direction
             in itertools.product(implementations, roofline_configs, directions)]

    bench_suite = TensorProductSuite(
        correctness_threshold = 5e-5,
        num_warmup=100,
        num_iter=100,
        bench_batch_size=200000,
        prng_seed=11111,
        torch_op=False,
        test_name="roofline")

    data_folder = bench_suite.run(tests, params.output_folder)

    if params.plot:
        plot({"data_folder": data_folder})
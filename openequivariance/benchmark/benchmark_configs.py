from openequivariance.benchmark.tpp_creation_utils import FullyConnectedTPProblem as FCTPP

# source: https://github.com/e3nn/e3nn/blob/main/examples/tetris.py
# running tetris will output the layers. I've only extracted the fully connected layers here. 
e3nn_torch_tetris = [
    # 0th Layer 
    FCTPP("1x0e", "1x0e", "150x0e + 50x1o + 50x2e"), #sc 
    FCTPP("1x0e", "1x0e", "1x0e"), #lin1
    FCTPP("1x0e + 1x1o + 1x2e", "1x0e", "150x0e + 50x1o + 50x2e"), #lin2
    FCTPP("1x0e + 1x1o + 1x2e", "1x0e", "1x0e"), #alpha
    
    # 1st Layer
    FCTPP("50x0e + 50x1o + 50x2e", "1x0e", "250x0e + 50x1o + 50x1e + 50x2o + 50x2e"), #sc
    FCTPP("50x0e + 50x1o + 50x2e", "1x0e", "50x0e + 50x1o + 50x2e"), #lin1 
    # FCTPP("50x0e + 50x1o + 50x2e", "1x0e + 1x1o + 1x2e",  "150x0e + 200x1o + 100x1e + 100x2o + 200x2e"), #tp
    FCTPP("150x0e + 200x1o + 100x1e + 100x2o + 200x2e", "1x0e", "250x0e + 50x1o + 50x1e + 50x2o + 50x2e"), #lin2
    FCTPP("150x0e + 200x1o + 100x1e + 100x2o + 200x2e", "1x0e", "1x0e"), #alpha

    # 2nd Layer 
    FCTPP("50x0e + 50x1o + 50x1e + 50x2o + 50x2e", "1x0e", "50x0o + 250x0e + 50x1o + 50x1e + 50x2o + 50x2e"), #sc 
    FCTPP("50x0e + 50x1o + 50x1e + 50x2o + 50x2e", "1x0e", "50x0e + 50x1o + 50x1e + 50x2o + 50x2e"), #lin1
    FCTPP("100x0o + 150x0e + 300x1o + 250x1e + 250x2o + 300x2e", "1x0e", "50x0o + 250x0e + 50x1o + 50x1e + 50x2o + 50x2e"), #lin2
    FCTPP("100x0o + 150x0e + 300x1o + 250x1e + 250x2o + 300x2e", "1x0e", "1x0e"), #alpha 

    # 3rd Layer 
    FCTPP("50x0o + 50x0e + 50x1o + 50x1e + 50x2o + 50x2e", "1x0e", "1x0o + 6x0e"), #sc
    FCTPP("50x0o + 50x0e + 50x1o + 50x1e + 50x2o + 50x2e", "1x0e", "50x0o + 50x0e + 50x1o + 50x1e + 50x2o + 50x2e"), #lin1
    FCTPP("150x0o + 150x0e", "1x0e", "1x0o + 6x0e"), #lin2
    FCTPP("150x0o + 150x0e", "1x0e", "1x0e"), #alpha  
]   


# jax version can be found here, but doesn't directly translate 
# https://github.com/e3nn/e3nn-jax/blob/main/examples/tetris_point.py


# source: https://github.com/e3nn/e3nn/blob/f95297952303347a8a3cfe971efe449c710c43b2/examples/tetris_polynomial.py#L66-L68
e3nn_torch_tetris_polynomial = [
    FCTPP("1x0e + 1x1o + 1x2e + 1x3o", "1x0e + 1x1o + 1x2e + 1x3o", "64x0e + 24x1e + 24x1o + 16x2e + 16x2o", label="tetris-poly-1"), #tp1 
    FCTPP("64x0e + 24x1e + 24x1o + 16x2e + 16x2o", "1x0e + 1x1o + 1x2e + 1x3o", "0o + 6x0e",                 label="tetris-poly-2"), #tp2 
]

# https://github.com/gcorso/DiffDock/blob/b4704d94de74d8cb2acbe7ec84ad234c09e78009/models/tensor_layers.py#L299
# specific irreps come from vivek's communication with DiffDock team
diffdock_configs = [
    FCTPP("10x1o + 10x1e + 48x0e + 48x0o", "1x0e + 1x1o",        "10x1o + 10x1e + 48x0e + 48x0o", shared_weights=False, label='DiffDock-L=1'),
    FCTPP("10x1o + 10x1e + 48x0e + 48x0o", "1x0e + 1x1o + 1x2e", "10x1o + 10x1e + 48x0e + 48x0o", shared_weights=False, label='DiffDock-L=2'),
]
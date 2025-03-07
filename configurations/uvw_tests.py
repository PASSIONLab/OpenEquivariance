from openequivariance.benchmark.tpp_creation_utils import FullyConnectedTPProblem as FCTPP

__all__ = [
    "basic_fully_connected_problems",
    "increasing_multiplicity_fully_connected_problems",
    "full_size_uvw_case",
    "basic_multi_interaction_problems"
]

basic_fully_connected_problems = [
        FCTPP("1x1e", "1x1e", "1x1e"),
        FCTPP("1x1e", "1x1e", "2x1e"),
        FCTPP("1x1e", "2x1e", "1x1e"), 
        FCTPP("2x1e", "1x1e", "1x1e"),
        FCTPP("2x1e", "2x1e", "1x1e"),
        FCTPP("2x1e", "2x1e", "2x1e"),
        FCTPP("2x1e", "2x1e", "4x1e") 
    ]

increasing_multiplicity_fully_connected_problems = [
        FCTPP("2x1e", "2x1e", "2x1e"),
        FCTPP("4x1e", "4x1e", "4x1e"),
        FCTPP("8x1e", "8x1e", "8x1e"),
        FCTPP("16x1e", "16x1e", "16x1e"),
        FCTPP("32x1e", "32x1e", "32x1e"),
    ]

full_size_uvw_case = [
        FCTPP("32x1e", "32x1e", "32x1e"),
        FCTPP("32x2e", "32x2e", "32x2e"),
        FCTPP("32x3e", "32x3e", "32x3e"),
        FCTPP("32x4e", "32x4e", "32x4e"),
        FCTPP("32x5e", "32x5e", "32x5e"),
    ]

basic_multi_interaction_problems = [
        FCTPP("2x1e + 1x0e", "2x1e", "4x1e"),
        FCTPP("2x1e", "2x1e + 1x0e", "4x1e"),
        FCTPP("2x1e + 1x0e", "2x1e + 1x0e", "4x1e"),
        FCTPP("32x1e + 32x0e", "32x1e + 32x0e", "32x1e + 32x0e"),
    ]
import importlib


def load_utilities(backend):
    module = importlib.import_module(f"openequivariance.core.oeq_utilities_{backend}")
    return module.DeviceProp, module.GPUTimer

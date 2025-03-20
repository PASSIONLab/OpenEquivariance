from openequivariance.implementations.LoopUnrollTP import LoopUnrollTP

class TensorProduct(LoopUnrollTP):
    '''
    Dispatcher class that selects the right implementation based on problem
    configuration. Right now, it just subclasses LoopUnrollTP.
    '''
    def __init__(self, problem, torch_op=True):
        super().__init__(problem, torch_op)

    @staticmethod
    def name():
        return super().name() 
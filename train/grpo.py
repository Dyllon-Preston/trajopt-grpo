import torch

class GRPO():

    def __init__(
            self,
            group_size: int,
            epsilon: float,
            ref_model: torch.nn.Module,
            beta: float,
    ):
        
        self.group_size = group_size
        self.epsilon = epsilon
        self.ref_model = ref_model
        self.beta = beta

    def train(self):
        

        for i in range(self.group_size):

            

            if self.ref_model is not None:
                raise NotImplementedError("Reference model is not None. This is not yet implemented.")
            else:
                D_kl = 0
            
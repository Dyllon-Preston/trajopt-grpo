import os
import torch

def load_checkpoint(
        policy,
        buffer, 
        optimizer, 
        trainer,
        path):
    """
    
    Load checkpoint.

    Args:
        policy (torch.nn.Module): Policy network.
        buffer (Rollout_Buffer): Rollout buffer.
        optimizer (torch.optim.Optimizer): Optimizer.
        trainer (Trainer): Trainer.
        path (str): Path to the checkpoint.
    
    Returns:
        Epoch number.
    """

    policy_state_dict = torch.load(os.path.join(path, "policy.pt"), weights_only=True)
    optimizer_state_dict = torch.load(os.path.join(path, "optimizer.pt"), weights_only=True)

    policy.load_state_dict(policy_state_dict)
    epoch = buffer.load_reward(os.path.join(path, "reward.csv"))
    optimizer.load_state_dict(optimizer_state_dict)
    trainer.set_epoch(epoch)
    return epoch
"""torchrun initialization, gradient synchronization and metric aggregation."""
import os

import torch
import torch.distributed as dist


def initialize():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    return device


def rank_world():
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def average_gradients(parameters):
    """Reduce once per optimizer update, allowing unequal segment counts per rank."""
    _, world = rank_world()
    if world == 1:
        return
    for parameter in parameters:
        if not parameter.requires_grad:
            continue
        gradient = parameter.grad
        if gradient is None:
            gradient = torch.zeros_like(parameter)
        dist.all_reduce(gradient)
        parameter.grad = gradient / world


def aggregate_training_metrics(metrics):
    """Reduce equal-sized local training batches, not replicated validation data."""
    _, world = rank_world()
    if world == 1:
        return metrics
    keys = [key for key in (
        "loss", "actor_loss", "critic_loss", "reference_kl", "entropy",
        "mean_reward", "n_trajectories",
    ) if key in metrics]
    if not keys:
        return metrics
    device = (torch.device("cuda", torch.cuda.current_device())
              if dist.get_backend() == "nccl" else torch.device("cpu"))
    values = torch.tensor([metrics[key] for key in keys], dtype=torch.float64, device=device)
    dist.all_reduce(values)
    result = dict(metrics)
    for key, value in zip(keys, values.cpu().tolist()):
        result[key] = int(value) if key == "n_trajectories" else value / world
    return result


def barrier():
    if dist.is_initialized():
        dist.barrier()

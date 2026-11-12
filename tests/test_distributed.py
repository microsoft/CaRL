from datetime import timedelta
import json
from pathlib import Path
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from carl.distributed import aggregate_training_metrics
from carl.train import log


def synchronize(rank, store, output):
    from carl.distributed import average_gradients
    dist.init_process_group("gloo", rank=rank, world_size=2, init_method=store,
                            timeout=timedelta(seconds=30))
    try:
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        parameter.grad = torch.tensor([1.0 + 2 * rank])
        average_gradients([parameter])
        assert parameter.grad.item() == 2.0
        log(Path(output), 1, {
            "loss": 1.0 + 2 * rank, "mean_reward": float(rank), "n_trajectories": 2,
            "policy_version": 1, "optimizer_minibatches": 3,
        })
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(sys.platform == "win32", reason="Windows Gloo transport is unavailable in this PyTorch build")
def test_real_two_process_gradient_synchronization(tmp_path):
    mp.spawn(synchronize, args=((tmp_path / "store").as_uri(), str(tmp_path)), nprocs=2, join=True)
    record = json.loads((tmp_path / "metrics.jsonl").read_text())
    assert record == {"step": 1, "loss": 2.0, "mean_reward": 0.5, "n_trajectories": 4,
                      "policy_version": 1, "optimizer_minibatches": 3}


def test_single_process_logging_is_unchanged(monkeypatch, tmp_path, capsys):
    import carl.distributed as distributed
    monkeypatch.setattr(distributed, "rank_world", lambda: (0, 1))
    monkeypatch.setattr(distributed.dist, "all_reduce", lambda values: pytest.fail("No collective expected"))
    metrics = {"loss": 2.0, "n_trajectories": 3, "policy_version": 1}
    assert aggregate_training_metrics(metrics) is metrics
    log(tmp_path, 1, metrics)
    expected = {"step": 1, **metrics}
    assert json.loads((tmp_path / "metrics.jsonl").read_text()) == expected
    assert json.loads(capsys.readouterr().out) == expected


@pytest.mark.parametrize("rank", [0, 1])
def test_distributed_logs_reduce_on_every_rank_but_write_only_rank_zero(monkeypatch, tmp_path, capsys, rank):
    import carl.distributed as distributed
    import carl.train as train
    monkeypatch.setattr(distributed, "rank_world", lambda: (rank, 2))
    monkeypatch.setattr(train, "rank_world", distributed.rank_world)
    monkeypatch.setattr(distributed.dist, "get_backend", lambda: "gloo")
    keys = ["loss", "actor_loss", "critic_loss", "reference_kl", "entropy",
            "mean_reward", "n_trajectories"]
    values = [[1.0, 2.0, 4.0, 0.1, 0.4, 0.25, 2],
              [3.0, 6.0, 2.0, 0.3, 0.8, 0.75, 2]]
    metrics = dict(zip(keys, values[rank]))
    metrics.update(policy_version=7, optimizer_minibatches=3)
    original = dict(metrics)
    calls = []

    def all_reduce(tensor):
        assert tensor.device.type == "cpu" and tensor.dtype == torch.float64
        calls.append(tensor.tolist())
        tensor.add_(torch.tensor(values[1 - rank], dtype=torch.float64))

    monkeypatch.setattr(distributed.dist, "all_reduce", all_reduce)
    log(tmp_path, 7, metrics)
    assert calls == [values[rank]]
    assert metrics == original
    if rank == 0:
        record = json.loads((tmp_path / "metrics.jsonl").read_text())
        assert record == pytest.approx({
            "step": 7, "loss": 2.0, "actor_loss": 4.0, "critic_loss": 3.0,
            "reference_kl": 0.2, "entropy": 0.6, "mean_reward": 0.5,
            "n_trajectories": 4, "policy_version": 7, "optimizer_minibatches": 3,
        })
        assert type(record["n_trajectories"]) is int
        assert json.loads(capsys.readouterr().out) == record
    else:
        assert not (tmp_path / "metrics.jsonl").exists()
        assert capsys.readouterr().out == ""


def test_replicated_validation_diagnostics_are_not_reduced(monkeypatch):
    import carl.distributed as distributed
    monkeypatch.setattr(distributed, "rank_world", lambda: (0, 2))
    monkeypatch.setattr(distributed.dist, "all_reduce", lambda values: pytest.fail("No collective expected"))
    metrics = {"val_mse": 0.1, "ev": None, "competence_auc": 0.8, "n_boundaries": 12,
               "passed_gate": False, "missing_gate_evidence": ["ev"], "gate_scope": "boundary_calibration"}
    assert aggregate_training_metrics(metrics) is metrics


def test_nccl_metric_reduction_uses_current_cuda_device(monkeypatch):
    import carl.distributed as distributed
    monkeypatch.setattr(distributed, "rank_world", lambda: (0, 2))
    monkeypatch.setattr(distributed.dist, "get_backend", lambda: "nccl")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    tensor = torch.tensor
    devices = []

    def create_tensor(values, *, dtype, device):
        devices.append(device)
        return tensor(values, dtype=dtype)

    monkeypatch.setattr(torch, "tensor", create_tensor)
    monkeypatch.setattr(distributed.dist, "all_reduce", lambda values: values.mul_(2))
    assert aggregate_training_metrics({"loss": 2.0}) == {"loss": 2.0}
    assert devices == [torch.device("cuda", 3)]

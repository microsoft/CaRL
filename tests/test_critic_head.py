from types import SimpleNamespace

import pytest
import torch
from torch import nn

from carl.critic.head import ValueHead, ValueModel


class TinyBackbone(nn.Module):
    """Small causal representation with no external model dependency."""
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.embedding = nn.Embedding(16, 4)

    def forward(self, input_ids, attention_mask, position_ids):
        hidden = self.embedding(input_ids) * attention_mask.unsqueeze(-1)
        return SimpleNamespace(last_hidden_state=hidden.cumsum(dim=1))


def test_head_initial_values_and_supplied_head_are_preserved():
    head = ValueHead(4, mlp_hidden=3)
    torch.testing.assert_close(head(torch.randn(2, 4)), torch.full((2,), 0.5))
    with torch.no_grad():
        head.fc2.bias.fill_(0.25)
    backbone = TinyBackbone()
    value = ValueModel(backbone, head)
    assert value.backbone is backbone and value.head is head
    tokens = torch.tensor([[3, 4]])
    torch.testing.assert_close(value(tokens, torch.ones_like(tokens)),
                               torch.sigmoid(torch.tensor([0.25])))
    assert next(value.head.parameters()).dtype == torch.float32
    assert next(value.head.parameters()).device == next(backbone.parameters()).device


def test_last_real_token_selection_handles_left_and_right_padding():
    value = ValueModel(TinyBackbone()).eval()
    with torch.no_grad():
        value.head.fc2.weight.fill_(0.1)
        left = value(torch.tensor([[0, 3, 4]]), torch.tensor([[0, 1, 1]]))
        right = value(torch.tensor([[3, 4, 0]]), torch.tensor([[1, 1, 0]]))
        unpadded = value(torch.tensor([[3, 4]]), torch.tensor([[1, 1]]))
    torch.testing.assert_close(left, right)
    torch.testing.assert_close(left, unpadded)


def test_critic_updates_the_existing_backbone_after_zero_head_initialization():
    torch.manual_seed(7)
    backbone = TinyBackbone()
    value = ValueModel(backbone)
    optimizer = torch.optim.SGD(value.parameters(), lr=0.1)
    tokens = torch.tensor([[3, 4]])
    before = backbone.embedding.weight.detach().clone()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = (value(tokens, torch.ones_like(tokens)) - 1).square().mean()
        loss.backward()
        optimizer.step()
    assert not torch.equal(before, backbone.embedding.weight)
    assert value.backbone.embedding.weight is backbone.embedding.weight


@pytest.mark.parametrize("ids,mask", [
    ([[3, 4]], [[1]]),
    ([[3, 4]], [[0, 0]]),
    ([[3, 4]], [[1, 2]]),
    ([[]], [[]]),
])
def test_invalid_critic_inputs_fail_explicitly(ids, mask):
    value = ValueModel(TinyBackbone())
    with pytest.raises(ValueError):
        value(torch.tensor(ids, dtype=torch.long), torch.tensor(mask))


@pytest.mark.parametrize("size", [0, -1, True])
def test_invalid_head_dimensions_are_rejected(size):
    with pytest.raises(ValueError):
        ValueHead(size)

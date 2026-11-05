"""Boundary values from a shared LLM backbone and a two-layer sigmoid head."""
import torch
import torch.nn as nn


class ValueHead(nn.Module):
    def __init__(self, hidden_size: int, mlp_hidden: int | None = None):
        super().__init__()
        mlp_hidden = hidden_size if mlp_hidden is None else mlp_hidden
        if any(type(n) is not int or n < 1 for n in (hidden_size, mlp_hidden)):
            raise ValueError("Value head dimensions must be positive integers")
        self.fc1 = nn.Linear(hidden_size, mlp_hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_hidden, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, h):  # h: (B, H) - last-token hidden state
        return torch.sigmoid(self.fc2(self.act(self.fc1(h)))).squeeze(-1)


class ValueModel(nn.Module):
    """Attach a value head to an existing backbone without copying its weights."""

    def __init__(self, backbone: nn.Module, head: ValueHead | None = None):
        super().__init__()
        self.backbone = backbone
        self.head = head if head is not None else ValueHead(backbone.config.hidden_size)
        self.head.to(device=next(backbone.parameters()).device, dtype=torch.float32)

    def forward(self, input_ids, attention_mask):
        if (input_ids.ndim != 2 or input_ids.shape != attention_mask.shape
                or input_ids.size(0) == 0 or input_ids.size(1) == 0):
            raise ValueError("Critic inputs must be non-empty, matching (batch, sequence) tensors")
        if not torch.all((attention_mask == 0) | (attention_mask == 1)):
            raise ValueError("Critic attention_mask must contain only 0 and 1")
        real_tokens = attention_mask.bool()
        if not torch.all(real_tokens.any(dim=1)):
            raise ValueError("Every critic input must contain at least one real token")

        position_ids = attention_mask.long().cumsum(dim=1) - 1
        position_ids = position_ids.masked_fill(~real_tokens, 0)
        out = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
        )
        h = out.last_hidden_state                # (B, T, H)
        positions = torch.arange(input_ids.size(1), device=input_ids.device).expand_as(input_ids)
        idx = positions.masked_fill(~real_tokens, -1).amax(dim=1)
        h_last = h[torch.arange(h.size(0), device=h.device), idx]
        return self.head(h_last.float())          # cast to fp32 for stable sigmoid

import torch
import torch.nn as nn

from typing import Iterable, Iterator, Tuple


def params_ema_update(teacher_model: nn.Module, student_model: nn.Module,
                      tau: float) -> None:
    new_params = student_model.named_parameters()
    target_params = teacher_model.named_parameters()
    for (n, p), (n1, p1) in zip(target_params, new_params):
        if n != n1:
            raise ValueError(f"Unexpected parameter name: {n1}")
        p.data.copy_(tau * p.data + (1 - tau) * p1.data)


def tau_generator(*,
                  initial_step: int = 0,
                  min_value: float = .2,
                  max_value: float = .7,
                  increase_steps: int = 512) -> Iterator[float]:
    step = initial_step
    slope = (max_value - min_value) / increase_steps
    while True:
        if step < increase_steps:
            yield step * slope + min_value
            step += 1
        else:
            yield max_value


def gen_nopeek_mask(length):
    # mask = rearrange(torch.triu(torch.ones(length, length)) == 1, 'h w -> w h')
    # mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    return torch.triu(torch.full((length, length), float('-inf')), diagonal=1)


class PositionalEmbedding(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, seq_len: int, dim: int) -> None:
        super().__init__()
        pos_weights = torch.empty(1, seq_len, dim)
        nn.init.normal_(pos_weights, std=0.02)
        self.pos_embedding = nn.Parameter(pos_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input has shape `(batch_size, seq_len, emb_dim)`"""
        # print('x.shape: {} self.pos_embedding.shape: {}'.format(x.shape, self.pos_embedding.shape))
        return x + self.pos_embedding[:, :x.shape[1], :]

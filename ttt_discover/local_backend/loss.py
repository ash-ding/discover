import torch


def importance_sampling_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    ratio = torch.exp(new_logprobs - old_logprobs)
    masked_loss = ratio * advantages * mask
    denom = mask.sum().clamp(min=1.0)
    return -(masked_loss.sum() / denom)


def ppo_clip_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    ratio = torch.exp(new_logprobs - old_logprobs)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    surr1 = ratio * advantages
    surr2 = clipped_ratio * advantages
    masked_loss = torch.min(surr1, surr2) * mask
    denom = mask.sum().clamp(min=1.0)
    return -(masked_loss.sum() / denom)

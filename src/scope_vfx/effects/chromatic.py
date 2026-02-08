import math

import torch


def chromatic_aberration(
    frames: torch.Tensor,
    intensity: float = 0.3,
    angle: float = 0.0,
) -> torch.Tensor:
    """Displace RGB channels in opposite directions for a chromatic aberration look.

    Args:
        frames: (T, H, W, C) tensor in [0, 1] range.
        intensity: Displacement strength (0-1, maps to 0-20 pixels).
        angle: Direction of displacement in degrees (0 = horizontal right).

    Returns:
        (T, H, W, C) tensor with displaced R and B channels.
    """
    if intensity <= 0:
        return frames

    max_shift = int(intensity * 20)
    if max_shift == 0:
        return frames

    rad = math.radians(angle)
    dx = int(round(max_shift * math.cos(rad)))
    dy = int(round(max_shift * math.sin(rad)))

    if dx == 0 and dy == 0:
        return frames

    result = frames.clone()

    # Red channel shifts one direction
    result[..., 0] = torch.roll(frames[..., 0], shifts=(dy, dx), dims=(1, 2))
    # Blue channel shifts the opposite direction
    result[..., 2] = torch.roll(frames[..., 2], shifts=(-dy, -dx), dims=(1, 2))
    # Green channel stays centred

    return result

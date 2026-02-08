import torch


def vhs_retro(
    frames: torch.Tensor,
    scan_line_intensity: float = 0.3,
    scan_line_count: int = 100,
    noise: float = 0.1,
    tracking: float = 0.2,
) -> torch.Tensor:
    """Apply a VHS / retro CRT look: scan lines, analog noise, and tracking distortion.

    Args:
        frames: (T, H, W, C) tensor in [0, 1] range.
        scan_line_intensity: Darkness of scan lines (0 = invisible, 1 = black).
        scan_line_count: Number of scan line pairs across the frame height.
        noise: Analog grain amount (0 = clean, 1 = heavy grain).
        tracking: Horizontal tracking distortion strength (0 = none, 1 = heavy).

    Returns:
        (T, H, W, C) tensor with the combined VHS effect applied.
    """
    _T, H, W, _C = frames.shape
    result = frames.clone()

    # --- Scan lines ---
    if scan_line_intensity > 0 and scan_line_count > 0:
        # Build a 1-D brightness mask that repeats `scan_line_count` times
        rows = torch.arange(H, device=frames.device, dtype=torch.float32)
        # sine wave creates smooth dark/light banding
        wave = torch.sin(rows * (scan_line_count * 3.14159 / H))
        # map [-1,1] -> [1-intensity, 1]  (dark bands at wave troughs)
        mask = 1.0 - scan_line_intensity * 0.5 * (1.0 - wave)
        result = result * mask.view(1, H, 1, 1)

    # --- Analog noise / film grain ---
    if noise > 0:
        grain = torch.randn_like(result) * (noise * 0.15)
        result = result + grain

    # --- Tracking distortion (horizontal sine-wave displacement) ---
    if tracking > 0:
        max_shift = tracking * 0.05  # fraction of image width
        rows_norm = torch.linspace(-1.0, 1.0, H, device=frames.device)
        # slowly varying sine gives the classic "wobbly VHS" look
        offsets = max_shift * torch.sin(rows_norm * 6.2832 * 3.0)

        # Build a sampling grid for grid_sample
        grid_y = torch.linspace(-1.0, 1.0, H, device=frames.device)
        grid_x = torch.linspace(-1.0, 1.0, W, device=frames.device)
        gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")

        # Shift x-coordinates per row
        gx = gx + offsets.view(H, 1)

        grid = torch.stack([gx, gy], dim=-1)  # (H, W, 2)
        grid = grid.unsqueeze(0).expand(result.shape[0], -1, -1, -1)  # (T, H, W, 2)

        # grid_sample expects (N, C, H, W), so permute
        result_nchw = result.permute(0, 3, 1, 2)
        result_nchw = torch.nn.functional.grid_sample(
            result_nchw, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        result = result_nchw.permute(0, 2, 3, 1)

    return result.clamp(0, 1)

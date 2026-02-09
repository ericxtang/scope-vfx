import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Curated Warhol palettes — 6 palettes x 8 colours ordered dark → light.
# Sub-sampled at runtime to match the active posterize level.
# ---------------------------------------------------------------------------

_PALETTE_DATA: list[list[tuple[float, float, float]]] = [
    # 0 – "Marilyn": hot pink / turquoise / yellow (more screenprint)
    [
        (0.06, 0.03, 0.08),  # ink black (purple-tinted)
        (0.38, 0.06, 0.44),  # deep violet
        (0.90, 0.10, 0.55),  # hot magenta
        (0.98, 0.42, 0.10),  # orange punch
        (0.99, 0.90, 0.06),  # acid yellow
        (0.05, 0.80, 0.78),  # turquoise
        (0.25, 0.95, 0.55),  # neon mint/green
        (0.98, 0.96, 0.86),  # warm paper white
    ],

    # 1 – "Electric": neon / fluorescent (more extreme contrast)
    [
        (0.02, 0.02, 0.09),  # ink navy-black
        (0.30, 0.00, 0.85),  # electric violet
        (0.96, 0.00, 0.62),  # fluorescent fuchsia
        (0.10, 0.98, 0.22),  # neon green
        (0.00, 0.70, 1.00),  # cyan-blue
        (1.00, 0.96, 0.05),  # neon yellow
        (1.00, 0.45, 0.00),  # blaze orange
        (0.98, 0.90, 0.96),  # pink-tinted paper
    ],

    # 2 – "Banana": Warhol banana vibe (less earthy, more pop)
    [
        (0.05, 0.05, 0.06),  # ink black
        (0.22, 0.16, 0.55),  # indigo
        (0.55, 0.10, 0.10),  # pop red
        (0.98, 0.84, 0.05),  # banana yellow
        (0.95, 0.55, 0.05),  # orange
        (0.10, 0.75, 0.65),  # teal
        (0.70, 0.95, 0.15),  # chartreuse
        (0.98, 0.96, 0.88),  # warm paper white
    ],

    # 3 – "Campbell's": red / white / gold (cleaner inks, less gray)
    [
        (0.05, 0.05, 0.06),  # ink black
        (0.55, 0.00, 0.10),  # deep crimson
        (0.95, 0.05, 0.08),  # bright can red
        (0.98, 0.78, 0.06),  # gold-yellow
        (0.05, 0.40, 0.95),  # cobalt accent (Warhol-ish twist)
        (0.95, 0.95, 0.92),  # warm off-white
        (0.80, 0.80, 0.78),  # screen gray
        (0.99, 0.98, 0.90),  # paper white
    ],

    # 4 – "Mao": bold political poster (hard primaries + skin/paper)
    [
        (0.06, 0.04, 0.03),  # warm ink black
        (0.10, 0.25, 0.75),  # poster blue
        (0.85, 0.05, 0.08),  # propaganda red
        (0.98, 0.85, 0.05),  # imperial yellow
        (0.00, 0.65, 0.55),  # teal green
        (0.98, 0.55, 0.12),  # orange flesh-pop
        (0.95, 0.82, 0.70),  # peach/skin paper
        (0.99, 0.97, 0.88),  # warm white
    ],

    # 5 – "Flowers": Warhol flowers (acid greens + candy petals)
    [
        (0.03, 0.06, 0.05),  # ink black-green
        (0.00, 0.55, 0.20),  # deep green
        (0.65, 0.98, 0.05),  # acid chartreuse
        (0.05, 0.70, 1.00),  # cyan
        (0.98, 0.20, 0.55),  # hot pink
        (0.98, 0.55, 0.05),  # orange
        (1.00, 0.90, 0.08),  # lemon yellow
        (0.99, 0.95, 0.88),  # paper white
    ],
]


_PALETTES = torch.tensor(_PALETTE_DATA, dtype=torch.float32)  # (6, 8, 3)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gaussian_blur_nchw(
    x: torch.Tensor, kernel_size: int = 5, sigma: float = 1.5,
) -> torch.Tensor:
    """Separable Gaussian blur on an (N, C, H, W) tensor."""
    k = kernel_size // 2
    coords = torch.arange(kernel_size, device=x.device, dtype=x.dtype) - k
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    g = g / g.sum()

    C = x.shape[1]
    # Horizontal pass
    kh = g.view(1, 1, 1, kernel_size).expand(C, -1, -1, -1)
    out = F.conv2d(F.pad(x, (k, k, 0, 0), mode="replicate"), kh, groups=C)
    # Vertical pass
    kv = g.view(1, 1, kernel_size, 1).expand(C, -1, -1, -1)
    out = F.conv2d(F.pad(out, (0, 0, k, k), mode="replicate"), kv, groups=C)
    return out


def _sobel_edges(luma: torch.Tensor) -> torch.Tensor:
    """Sobel edge magnitude from (N, 1, H, W) luminance."""
    sx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        device=luma.device, dtype=luma.dtype,
    ).view(1, 1, 3, 3)
    sy = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        device=luma.device, dtype=luma.dtype,
    ).view(1, 1, 3, 3)
    p = F.pad(luma, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(p, sx)
    gy = F.conv2d(p, sy)
    return torch.sqrt(gx * gx + gy * gy)


# ---------------------------------------------------------------------------
# Public effect
# ---------------------------------------------------------------------------

def warhol(
    frames: torch.Tensor,
    palette: int = 0,
    posterize: int = 4,
    ink: float = 0.7,
    edge_thresh: float = 0.15,
) -> torch.Tensor:
    """Apply a Warhol screen-print effect.

    Pipeline:
      1. Gaussian pre-blur for temporal stability
      2. Sobel edge extraction → binary ink mask
      3. Luminance posterization into flat bands (segmentation-lite)
      4. Band → curated palette colour remap
      5. Ink-edge overlay

    Args:
        frames: (T, H, W, 3) tensor in [0, 1] range.
        palette: Palette index 0-5
                 (0 Marilyn, 1 Electric, 2 Banana, 3 Campbell's, 4 Mao, 5 Flowers).
        posterize: Number of luminance bands / colour levels (2-8).
        ink: Opacity of the ink-edge overlay (0 = no outlines, 1 = solid black).
        edge_thresh: Sobel threshold (higher = fewer / only strongest edges).

    Returns:
        (T, H, W, 3) tensor with the Warhol effect applied.
    """
    T, H, W, _C = frames.shape
    device = frames.device

    # Work in NCHW for conv operations
    nchw = frames.permute(0, 3, 1, 2)  # (T, 3, H, W)

    # ---- 1. Pre-blur (temporal stability) --------------------------------
    blurred = _gaussian_blur_nchw(nchw, kernel_size=5, sigma=1.5)

    # BT.601 luminance from the blurred image
    luma = (
        0.299 * blurred[:, 0:1]
        + 0.587 * blurred[:, 1:2]
        + 0.114 * blurred[:, 2:3]
    )  # (T, 1, H, W)

    # ---- 2. Ink-edge extraction ------------------------------------------
    edge_mag = _sobel_edges(luma)
    # Per-frame normalisation keeps the threshold content-independent
    e_max = edge_mag.flatten(1).max(dim=1).values.view(T, 1, 1, 1).clamp(min=1e-5)
    edge_norm = edge_mag / e_max
    ink_mask = (edge_norm > edge_thresh).float()
    # Dilate for thicker screen-print ink lines
    ink_mask = F.max_pool2d(ink_mask, kernel_size=3, stride=1, padding=1)

    # ---- 3. Posterize / segment luminance into flat bands ----------------
    n_levels = max(int(posterize), 2)
    band_idx = (luma * (n_levels - 1)).round().long().clamp(0, n_levels - 1)

    # ---- 4. Palette colour remap ----------------------------------------
    pal_idx = max(0, min(int(palette), len(_PALETTE_DATA) - 1))
    pal_8 = _PALETTES[pal_idx].to(device=device, dtype=frames.dtype)  # (8, 3)

    if n_levels >= 8:
        colours = pal_8[:n_levels]
    else:
        sample_indices = torch.linspace(0, 7, n_levels, device=device).long()
        colours = pal_8[sample_indices]  # (n_levels, 3)

    flat_idx = band_idx.squeeze(1).reshape(-1)  # (T*H*W,)
    result = colours[flat_idx].view(T, H, W, 3)

    # ---- 5. Ink overlay --------------------------------------------------
    if ink > 0:
        ink_hw = ink_mask.squeeze(1).unsqueeze(-1)  # (T, H, W, 1)
        result = result * (1.0 - ink * ink_hw)

    return result.clamp(0, 1)

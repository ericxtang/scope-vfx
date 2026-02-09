from typing import TYPE_CHECKING

import torch

from scope.core.pipelines.interface import Pipeline, Requirements

from .effects import chromatic_aberration, vhs_retro, warhol
from .schema import VFXConfig

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig


class VFXPipeline(Pipeline):
    """GPU-accelerated visual effects pipeline.

    Chains multiple effects (chromatic aberration, VHS/retro CRT, ...) on
    incoming video frames.  Every parameter is a runtime slider so effects
    can be tweaked live during streaming.
    """

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return VFXConfig

    def __init__(self, device: torch.device | None = None, **kwargs):
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def prepare(self, **kwargs) -> Requirements:
        """We need exactly one input frame per call."""
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        """Apply enabled effects to input video frames.

        Args:
            video: List of input frame tensors, each (1, H, W, C) in [0, 255].

        Returns:
            Dict with ``"video"`` key containing processed frames in [0, 1] range.
        """
        video = kwargs.get("video")
        if video is None:
            raise ValueError("VFXPipeline requires video input")

        # Stack input frames -> (T, H, W, C) and normalise to [0, 1]
        frames = torch.stack([frame.squeeze(0) for frame in video], dim=0)
        frames = frames.to(device=self.device, dtype=torch.float32) / 255.0

        # --- Effect chain (order matters) ---

        if kwargs.get("chromatic_enabled", True):
            frames = chromatic_aberration(
                frames,
                intensity=kwargs.get("chromatic_intensity", 0.3),
                angle=kwargs.get("chromatic_angle", 0.0),
            )

        if kwargs.get("vhs_enabled", False):
            frames = vhs_retro(
                frames,
                scan_line_intensity=kwargs.get("scan_line_intensity", 0.3),
                scan_line_count=kwargs.get("scan_line_count", 100),
                noise=kwargs.get("vhs_noise", 0.1),
                tracking=kwargs.get("tracking_distortion", 0.2),
            )

        if kwargs.get("warhol_enabled", False):
            frames = warhol(
                frames,
                palette=kwargs.get("warhol_palette", 0),
                posterize=kwargs.get("warhol_posterize", 4),
                ink=kwargs.get("warhol_ink", 0.7),
                edge_thresh=kwargs.get("warhol_edge_thresh", 0.15),
            )

        return {"video": frames.clamp(0, 1)}

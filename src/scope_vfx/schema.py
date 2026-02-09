from pydantic import Field

from scope.core.pipelines.base_schema import BasePipelineConfig, ModeDefaults, UsageType, ui_field_config


class VFXConfig(BasePipelineConfig):
    """Configuration for the VFX Pack pipeline."""

    pipeline_id = "vfx-pack"
    pipeline_name = "VFX Pack"
    pipeline_description = "GPU-accelerated visual effects: chromatic aberration, VHS/retro CRT, and more"

    supports_prompts = False

    usage = [UsageType.POSTPROCESSOR]
    modes = {"video": ModeDefaults(default=True)}

    # --- Chromatic Aberration ---

    chromatic_enabled: bool = Field(
        default=True,
        description="Enable chromatic aberration (RGB channel displacement)",
        json_schema_extra=ui_field_config(order=1, label="Chromatic Aberration"),
    )

    chromatic_intensity: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Strength of the RGB channel displacement (0 = none, 1 = maximum)",
        json_schema_extra=ui_field_config(order=2, label="Intensity"),
    )

    chromatic_angle: float = Field(
        default=0.0,
        ge=0.0,
        le=360.0,
        description="Direction of the channel displacement in degrees",
        json_schema_extra=ui_field_config(order=3, label="Angle"),
    )

    # --- VHS / Retro CRT ---

    vhs_enabled: bool = Field(
        default=False,
        description="Enable VHS / retro CRT effect (scan lines, noise, tracking)",
        json_schema_extra=ui_field_config(order=10, label="VHS / Retro CRT"),
    )

    scan_line_intensity: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Darkness of the scan lines (0 = invisible, 1 = fully black)",
        json_schema_extra=ui_field_config(order=11, label="Scan Lines"),
    )

    scan_line_count: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Number of scan lines across the frame height",
        json_schema_extra=ui_field_config(order=12, label="Line Count"),
    )

    vhs_noise: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Amount of analog noise / film grain",
        json_schema_extra=ui_field_config(order=13, label="Noise"),
    )

    tracking_distortion: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Horizontal tracking distortion (wavy displacement)",
        json_schema_extra=ui_field_config(order=14, label="Tracking"),
    )

    # --- Warhol Pop-Art ---

    warhol_enabled: bool = Field(
        default=False,
        description="Enable Warhol screen-print effect (ink edges, flat palette colours)",
        json_schema_extra=ui_field_config(order=20, label="Warhol Pop-Art"),
    )

    warhol_palette: int = Field(
        default=0,
        ge=0,
        le=5,
        description="Colour palette (0 Marilyn, 1 Electric, 2 Banana, 3 Campbell's, 4 Mao, 5 Flowers)",
        json_schema_extra=ui_field_config(order=21, label="Palette"),
    )

    warhol_posterize: int = Field(
        default=4,
        ge=2,
        le=8,
        description="Number of flat colour bands (fewer = bolder screen-print look)",
        json_schema_extra=ui_field_config(order=22, label="Color Levels"),
    )

    warhol_ink: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Ink-edge overlay strength (0 = no outlines, 1 = solid black)",
        json_schema_extra=ui_field_config(order=23, label="Ink Strength"),
    )

    warhol_edge_thresh: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Edge detection threshold (higher = fewer, bolder edges)",
        json_schema_extra=ui_field_config(order=24, label="Edge Threshold"),
    )

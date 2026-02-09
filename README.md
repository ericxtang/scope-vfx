# scope-vfx

GPU-accelerated visual effects pack for [Daydream Scope](https://github.com/daydreamlive/scope).

Applies real-time video effects to any camera or video input — chromatic aberration, VHS / retro CRT, and more. Every parameter is a live slider so you can tweak effects during streaming.

## Effects

| Effect | Description |
|--------|-------------|
| **Chromatic Aberration** | Displaces RGB channels in opposite directions for a cinematic / glitch look |
| **VHS / Retro CRT** | Scan lines, analog noise, and tracking distortion for a nostalgic analog feel |
| **Warhol Pop Art** | Saturated colors, high contrast, pop art style |


More effects coming soon.

## Install

In Scope, open **Settings > Plugins** and enter:

```
git+https://github.com/viborc/scope-vfx.git
```

Or install from a local directory during development:

```
/path/to/scope-vfx
```

## Development

1. Clone this repo
2. In Scope, install the plugin using the local path to the repo root
3. Edit effect files in `src/scope_vfx/effects/`
4. Click **Reload** next to the plugin in Settings to pick up changes

### Adding a new effect

1. Create a new file in `src/scope_vfx/effects/` (e.g. `glitch.py`)
2. Write a function that takes `(frames: torch.Tensor, ...) -> torch.Tensor`
3. Add its parameters to `schema.py`
4. Wire it into the effect chain in `pipeline.py`
5. Re-export from `effects/__init__.py`

## Changelog

### 0.1.1
- Registered as a post-processor instead of a main pipeline — VFX Pack now chains after any generative model (LongLive, StreamDiffusion, etc.) to apply effects to AI-generated video output

### 0.1.0
- Initial release with chromatic aberration and VHS/retro CRT effects

## Requirements

- Daydream Scope (any version with plugin support)
- No additional dependencies — uses only PyTorch (provided by Scope)

# ComfyUI-ScreenRemoval

A ComfyUI extension for automatically removing patio screens, window screens, and other regular mesh patterns from photographs.

## Overview

ComfyUI-ScreenRemoval is designed for wildlife, bird, and nature photographers who frequently photograph through patio enclosures or insect screens.

Unlike traditional AI inpainting workflows, this project uses a hybrid restoration pipeline:

1. Detect the screen mesh using frequency-domain analysis (FFT).
2. Suppress the mesh while preserving scene detail.
3. Generate a residual mask for damaged regions.
4. Repair only those regions with FLUX Fill or SDXL Inpainting.

The goal is to reconstruct the original image while minimizing AI hallucination.

## Planned Features

- FFT-based screen calibration
- Automatic screen profile generation
- Hybrid mathematical + AI screen removal
- FLUX Fill integration
- SDXL Inpainting support
- Batch folder processing
- Calibration profiles for fixed patio screens
- ComfyUI Manager compatible
- GPU accelerated
- Open source (MIT)

## Project Status

🚧 Early Development

Current milestone:

- [ ] Project skeleton
- [ ] Screen Calibration node
- [ ] FFT visualization
- [ ] Screen suppression
- [ ] Residual detection
- [ ] FLUX integration
- [ ] Batch processing
- [ ] Version 1.0 release

## Example Workflow

Input Image

↓

Screen Calibration

↓

Screen Suppression

↓

Residual Mask

↓

FLUX Fill

↓

Save Image

## License

MIT License

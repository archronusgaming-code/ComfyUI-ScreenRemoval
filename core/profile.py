"""
core/profile.py

Calibration profile datatypes.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass
class ScreenProfile:
    schema: int = 1

    width: int = 0
    height: int = 0

    pitch_x: float = 0.0
    pitch_y: float = 0.0

    rotation: float = 0.0

    confidence: float = 0.0

    screen_type: str = "square_mesh"

    def to_dict(self):
        return asdict(self)

    def save(self, filename: str | Path):

        path = Path(filename)

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                self.to_dict(),
                f,
                indent=4,
            )

    @classmethod
    def load(cls, filename):

        with open(filename, "r", encoding="utf-8") as f:

            data = json.load(f)

        return cls(**data)
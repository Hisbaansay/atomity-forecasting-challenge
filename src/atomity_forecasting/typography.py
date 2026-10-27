"""Locate locally installed fonts without redistributing licensed font files."""
from __future__ import annotations

import os
from pathlib import Path


def times_new_roman_files() -> dict[str, Path] | None:
    """Find a complete Times New Roman family, or return None for a fallback.

    Set ATOMITY_FONT_DIR to an installed font directory on another machine.
    Font binaries are never copied into the project or submission archive.
    """
    configured = os.environ.get("ATOMITY_FONT_DIR")
    roots = [Path(configured)] if configured else [
        Path("/System/Library/Fonts/Supplemental"), Path("/Library/Fonts"),
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts",
        Path("/usr/share/fonts/truetype/msttcorefonts"),
        Path.home() / "Library/Fonts", Path.home() / ".local/share/fonts",
    ]
    families = [
        ("Times New Roman.ttf", "Times New Roman Bold.ttf", "Times New Roman Italic.ttf", "Times New Roman Bold Italic.ttf"),
        ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"),
        ("Times_New_Roman.ttf", "Times_New_Roman_Bold.ttf", "Times_New_Roman_Italic.ttf", "Times_New_Roman_Bold_Italic.ttf"),
    ]
    for root in roots:
        for names in families:
            paths = dict(zip(("regular", "bold", "italic", "bold_italic"), (root / name for name in names)))
            if all(path.is_file() for path in paths.values()):
                return paths
    if configured:
        raise ValueError("ATOMITY_FONT_DIR must contain all four Times New Roman styles")
    return None

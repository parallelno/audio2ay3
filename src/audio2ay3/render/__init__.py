"""Emulate YM register streams to audio."""

from __future__ import annotations

from .audio_out import write_audio, write_mp3, write_wav
from .renderer import Renderer

__all__ = ["Renderer", "write_audio", "write_wav", "write_mp3"]

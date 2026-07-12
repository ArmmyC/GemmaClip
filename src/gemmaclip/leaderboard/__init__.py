"""CLI-only Fireworks leaderboard pipeline.

This package is deliberately independent from the public web provider and its
routed Gemma services.  The competition entrypoint imports it only when
``GEMMACLIP_PROVIDER=fireworks_leaderboard`` is selected.
"""

from gemmaclip.leaderboard.config import (
    DEFAULT_FIREWORKS_LEADERBOARD_FALLBACK_MODEL,
    DEFAULT_FIREWORKS_LEADERBOARD_GENERATION_MODEL,
    DEFAULT_FIREWORKS_LEADERBOARD_REVIEW_MODEL,
    FireworksLeaderboardConfig,
    load_fireworks_leaderboard_config,
)
from gemmaclip.leaderboard.pipeline import (
    build_leaderboard_fallback_captions,
    generate_fireworks_leaderboard_captions,
)

__all__ = [
    "DEFAULT_FIREWORKS_LEADERBOARD_FALLBACK_MODEL",
    "DEFAULT_FIREWORKS_LEADERBOARD_GENERATION_MODEL",
    "DEFAULT_FIREWORKS_LEADERBOARD_REVIEW_MODEL",
    "FireworksLeaderboardConfig",
    "build_leaderboard_fallback_captions",
    "generate_fireworks_leaderboard_captions",
    "load_fireworks_leaderboard_config",
]

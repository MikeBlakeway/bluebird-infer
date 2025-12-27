"""G2P alignment scaffold.

Provides a placeholder function to align lyrics to phoneme timings.
Will be replaced with a real G2P alignment and duration model.
"""

from typing import List, Dict, Any


def align_lyrics_to_phonemes(lyrics: List[str], bpm: int) -> Dict[str, Any]:
    """Stub alignment returning naive per-line timings.

    Each line receives an equal slice of time; phonemes are not computed.
    """
    if not lyrics:
        return {"lines": []}
    # Assume 1 bar per line at given BPM, 4/4 time
    seconds_per_beat = 60.0 / max(1, bpm)
    line_duration = seconds_per_beat * 4
    lines = []
    start = 0.0
    for text in lyrics:
        lines.append({"text": text, "start": start, "duration": line_duration})
        start += line_duration
    return {"lines": lines, "total": start}

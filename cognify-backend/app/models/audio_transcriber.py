import os
import re
from typing import Dict, List, Tuple
from faster_whisper import WhisperModel

_whisper = WhisperModel("small", device="cpu", compute_type="int8")

def transcribe_with_conf(audio_path: str) -> Tuple[str, float, List[Tuple[str, float]], float]:
    """
    Returns:
      transcript, avg_confidence, list of (word, prob), duration_sec
    """
    segments, info = _whisper.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        word_timestamps=True
    )

    words: List[Tuple[str, float]] = []
    text_parts: List[str] = []

    for seg in segments:
        # seg.text is sentence-level text
        if seg.text:
            text_parts.append(seg.text.strip())
        # Per-word confidences
        if seg.words:
            for w in seg.words:
                # w.probability can be None; guard with default
                prob = float(w.probability) if w.probability is not None else 1.0
                word = w.word.strip()
                if word:
                    words.append((word, prob))

    transcript = " ".join(text_parts).strip()
    if not words:
        return transcript, 0.0, [], float(info.duration)

    avg_conf = sum(p for _, p in words) / len(words)
    return transcript, float(avg_conf), words, float(info.duration)

def detect_broken_words(words: List[Tuple[str, float]], prob_threshold: float = 0.55) -> List[str]:
    """Return words with low probability (likely misheard or 'broken')."""
    broken = []
    for w, p in words:
        # filter out punctuation-ish tokens Whisper sometimes emits
        if re.fullmatch(r"[\W_]+", w):
            continue
        if p < prob_threshold:
            # strip punctuation at ends
            cleaned = re.sub(r"(^\W+|\W+$)", "", w)
            if cleaned:
                broken.append(cleaned)
    # dedup while preserving order
    seen = set()
    out = []
    for b in broken:
        if b.lower() not in seen:
            seen.add(b.lower())
            out.append(b)
    return out

def summarize_transcript_simple(text: str, max_words: int = 24) -> str:
    """Super-simple 'summary': first sentence or first N words."""
    text = text.strip()
    if not text:
        return ""
    # Prefer first sentence
    parts = re.split(r"(?<=[\.\!\?])\s+", text)
    if parts:
        first = parts[0].strip()
        if len(first.split()) <= max_words:
            return first
    # Fallback to first N words
    return " ".join(text.split()[:max_words])

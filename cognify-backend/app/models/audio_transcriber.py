import os
import re
from typing import Dict, List, Tuple, Optional
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr

from faster_whisper import WhisperModel

# ---------- Whisper ----------
_whisper = WhisperModel("small", device="cpu", compute_type="int8")

def transcribe_with_conf(audio_path: str):
    """
    Returns:
      transcript: str
      avg_confidence: float
      words: list[dict] -> [{word, start, end, prob}]
      duration_sec: float
      segments: list[dict] -> [{start, end, text, avg_conf}]
    """
    segments, info = _whisper.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        word_timestamps=True
    )

    words = []
    seg_out = []
    text_parts: List[str] = []

    for seg in segments:
        # segment text + confidence
        seg_probs = []
        if seg.words:
            for w in seg.words:
                prob = float(w.probability) if w.probability is not None else 1.0
                words.append({
                    "word": (w.word or "").strip(),
                    "start": float(w.start or seg.start),
                    "end": float(w.end or seg.end),
                    "prob": prob,
                })
                seg_probs.append(prob)
        seg_text = (seg.text or "").strip()
        if seg_text:
            text_parts.append(seg_text)
            avgc = float(np.mean(seg_probs)) if seg_probs else 1.0
            seg_out.append({
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg_text,
                "avg_conf": avgc
            })

    transcript = " ".join([t for t in text_parts if t]).strip()
    duration = float(get_audio_duration(audio_path))
    avg_conf = float(np.mean([w["prob"] for w in words])) if words else 0.0
    return transcript, avg_conf, words, duration, seg_out


def detect_broken_words(words: List[Dict], prob_threshold: float = 0.55):
    """Return low-probability words + contiguous spans with timestamps."""
    broken_tokens = []
    for w in words:
        token = w["word"]
        if not token or re.fullmatch(r"[\W_]+", token):  # punctuation-ish
            continue
        if w["prob"] < prob_threshold:
            broken_tokens.append(w)

    # Dedup simple word list
    seen = set()
    broken_word_list = []
    for w in broken_tokens:
        low = w["word"].lower()
        if low and low not in seen:
            seen.add(low)
            broken_word_list.append(w["word"])

    # Merge to contiguous spans for visualization
    spans = merge_low_conf_spans(broken_tokens)
    return broken_word_list, spans


def merge_low_conf_spans(tokens: List[Dict], max_gap: float = 0.25):
    """
    Merge low-conf tokens into spans when they are close in time.
    Returns [{start, end, words:[...]}]
    """
    if not tokens:
        return []
    tokens = sorted(tokens, key=lambda x: x["start"])
    spans = []
    cur = {"start": tokens[0]["start"], "end": tokens[0]["end"], "words": [tokens[0]["word"]]}

    for t in tokens[1:]:
        gap = t["start"] - cur["end"]
        if gap <= max_gap:
            cur["end"] = max(cur["end"], t["end"])
            cur["words"].append(t["word"])
        else:
            spans.append(cur)
            cur = {"start": t["start"], "end": t["end"], "words": [t["word"]]}
    spans.append(cur)
    return spans


def summarize_transcript_simple(text: str, max_words: int = 24) -> str:
    text = text.strip()
    if not text:
        return ""
    parts = re.split(r"(?<=[\.\!\?])\s+", text)
    if parts:
        first = parts[0].strip()
        if len(first.split()) <= max_words:
            return first
    return " ".join(text.split()[:max_words])


# ---------- Utility ----------
def get_audio_duration(path: str) -> float:
    try:
        import soundfile as _sf
        info = _sf.info(path)
        return float(info.duration)
    except Exception:
        y, sr = librosa.load(path, sr=None, mono=True)
        return float(len(y) / sr)


# ---------- Simple "Repair": Noise Reduction + export ----------
def denoise_audio(input_path: str, out_path: str) -> Dict:
    """
    Basic noise reduction (stationary) + save cleaned WAV.
    Returns metadata with SNR estimate deltas.
    """
    y, sr = librosa.load(input_path, sr=None, mono=True)
    # crude noise profile: first 0.5s (or less if shorter)
    head = y[: int(min(0.5, len(y)/sr) * sr)]
    reduced = nr.reduce_noise(y=y, sr=sr, y_noise=head, prop_decrease=0.9)
    sf.write(out_path, reduced, sr)

    # naive SNR estimate (RMS)
    def snr(sig):
        rms = np.sqrt(np.mean(sig**2) + 1e-8)
        # treat noise as difference from a smoothed version
        import scipy.signal as sps
        sm = sps.medfilt(sig, kernel_size=99)
        noise = sig - sm
        nrms = np.sqrt(np.mean(noise**2) + 1e-8)
        return 20*np.log10(rms/nrms)

    try:
        import scipy  # only for the crude SNR
        before = snr(y)
        after = snr(reduced)
    except Exception:
        before = after = 0.0

    return {"sr": sr, "duration": len(y)/sr, "snr_before": before, "snr_after": after}


# ---------- "Embeddings" for audio labeling (MFCC mean) ----------
def audio_embedding_mfcc(path: str, n_mfcc: int = 40) -> List[float]:
    y, sr = librosa.load(path, sr=None, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    vec = np.mean(mfcc, axis=1)  # (n_mfcc,)
    return vec.astype(float).tolist()


# ---------- Diarization (fallback heuristic) ----------
def diarize_energy_fallback(path: str, top_db: float = 35.0, min_len: float = 0.7):
    """
    Energy-based segmentation fallback that pretends single speaker,
    splits by silence; returns [{start, end, speaker}]
    """
    y, sr = librosa.load(path, sr=None, mono=True)
    # non-silent intervals (start, end) in samples
    intervals = librosa.effects.split(y, top_db=top_db)
    out = []
    for i, (s, e) in enumerate(intervals):
        dur = (e - s) / sr
        if dur < min_len:
            continue
        out.append({
            "start": s / sr,
            "end": e / sr,
            "speaker": f"spk_1"
        })
    return out

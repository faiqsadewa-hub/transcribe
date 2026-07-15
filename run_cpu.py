"""Standalone CPU runner reusing transcribe.py's parameters (no torch dependency)."""
import sys, time, subprocess, warnings
from pathlib import Path
import numpy as np
import imageio_ffmpeg
from faster_whisper import WhisperModel

warnings.filterwarnings("ignore")

SRC = "/root/.claude/uploads/8e163ca6-e9bf-5ccc-aad8-7b4b2d6ba747/60a89f49-BCG_final_round_interview__Sunaryo_Gunawan.m4a"


def load_audio(file_path, start=None, dur=None):
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ff]
    if start is not None:
        cmd += ["-ss", str(start)]
    if dur is not None:
        cmd += ["-t", str(dur)]
    cmd += ["-i", str(file_path), "-f", "s16le", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1", "-v", "quiet", "pipe:1"]
    result = subprocess.run(cmd, capture_output=True)
    a = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return a


def transcribe(audio, model, initial_prompt=None):
    segs_gen, info = model.transcribe(
        audio, language="id", beam_size=5, temperature=0,
        condition_on_previous_text=False, initial_prompt=initial_prompt,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300, speech_pad_ms=300),
        no_speech_threshold=0.6, compression_ratio_threshold=2.0,
        word_timestamps=True,
    )
    out = []
    for seg in segs_gen:
        out.append({"start": seg.start, "end": seg.end, "text": seg.text.strip()})
    return out, info


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "bench"
    model_size = sys.argv[2] if len(sys.argv) > 2 else "small"
    print(f"Loading model {model_size} (cpu int8)...", flush=True)
    t0 = time.time()
    model = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=4)
    print(f"Model loaded in {time.time()-t0:.1f}s", flush=True)

    if mode == "bench":
        audio = load_audio(SRC, start=300, dur=60)  # 60s from 5:00
        t0 = time.time()
        segs, info = transcribe(audio, model)
        el = time.time() - t0
        print(f"[{model_size}] 60s audio -> {el:.1f}s  ({60/el:.2f}x realtime)  lang={info.language} p={info.language_probability:.2f}")
        for s in segs[:6]:
            print(f"  {s['start']:.1f}-{s['end']:.1f}: {s['text']}")

"""Minimal self-contained Whisper ONNX inference (multilingual small).

Uses ailia's whisper encoder/decoder ONNX graphs (data only) with original
whisper tokenizer assets from the openai-whisper sdist. Greedy decoding with
whisper timestamp rules, per-30s-window, seek by last timestamp.
"""
import base64
import sys
import time
import subprocess
from pathlib import Path

import numpy as np
import tiktoken
import onnxruntime as ort

import os
S = Path(os.environ.get("WHISPER_ONNX_DIR", "."))
ASSETS = Path(os.environ.get("WHISPER_ASSETS_DIR", str(S / "assets")))

SAMPLE_RATE = 16000
N_FFT = 400
HOP = 160
N_FRAMES = 3000  # 30 s of mel frames
N_AUDIO_PER_WINDOW = 30 * SAMPLE_RATE

SOT = 50258
EOT = 50257
TASK_TRANSCRIBE = 50359
NO_TIMESTAMPS = 50363
TS_BEGIN = 50364  # <|0.00|>
TS_END = 51864    # <|30.00|>
LANG_BASE = 50259
LANG_INDEX = {"en": 0, "id": 16}  # verified against whisper tokenizer.py

MAX_CTX = 451  # decoder kv_cache time dim


def build_tokenizer():
    ranks = {}
    for line in open(ASSETS / "multilingual.tiktoken"):
        if line.strip():
            tok, rank = line.split()
            ranks[base64.b64decode(tok)] = int(rank)
    return tiktoken.Encoding(
        name="whisper_multilingual", explicit_n_vocab=None,
        pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        mergeable_ranks=ranks, special_tokens={},
    )


def log_mel(audio: np.ndarray) -> np.ndarray:
    """Whisper-style 80-bin log-mel of a 30s (padded) chunk -> [80, 3000]."""
    filters = np.load(ASSETS / "mel_filters.npz")["mel_80"]  # [80, 201]
    window = np.hanning(N_FFT + 1)[:-1]
    pad = N_FFT // 2
    x = np.pad(audio, (pad, pad), mode="reflect")
    n_steps = 1 + (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_steps)[:, None]
    frames = x[idx] * window
    stft = np.fft.rfft(frames, axis=1)  # [T, 201]
    mag = (np.abs(stft) ** 2).T[:, :-1]  # drop last frame like torch.stft[..., :-1]
    mel = filters @ mag
    log_spec = np.log10(np.clip(mel, 1e-10, None))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).astype(np.float32)


# Byte-level suppress list: whisper's default non-speech tokens (subset built at runtime)
def non_speech_tokens(enc):
    symbols = list('"#()*+/:;<=>@[\\]^_`{|}~「」『』') + \
        "<< >> <<< >>> -- --- -( -[ (' (\" (( )) ((( ))) [[ ]] {{ }} ♪♪ ♪♪♪".split()
    miscellaneous = set("♩♪♫♬♭♮♯")
    result = set()
    for symbol in symbols + list(miscellaneous):
        for tok in [symbol, " " + symbol]:
            try:
                ids = enc.encode(tok)
                if len(ids) == 1:
                    result.add(ids[0])
            except Exception:
                pass
    # -1 defaults also include these two special-ish ids
    return sorted(result)


class WhisperONNX:
    def __init__(self, enc_path, dec_path, threads=4, kv_layers=24, n_state=768):
        self.kv_layers = kv_layers
        self.n_state = n_state
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.enc = ort.InferenceSession(str(enc_path), so, providers=["CPUExecutionProvider"])
        self.dec = ort.InferenceSession(str(dec_path), so, providers=["CPUExecutionProvider"])
        self.tok = build_tokenizer()
        self.suppress = non_speech_tokens(self.tok)

    def encode(self, mel):
        return self.enc.run(["audio_features"], {"mel": mel[None]})[0]

    def dec_step(self, tokens, audio_features, kv_cache, offset):
        logits, kv = self.dec.run(
            ["logits", "output_kv_cache"],
            {"tokens": np.asarray(tokens, dtype=np.int64),
             "audio_features": audio_features,
             "kv_cache": kv_cache,
             "offset": np.asarray(offset, dtype=np.int64)})
        return logits, kv

    def decode_window(self, audio_features, lang, max_tokens=224, temperature=0.0):
        initial = [SOT, LANG_BASE + LANG_INDEX[lang], TASK_TRANSCRIBE]
        kv = np.zeros((self.kv_layers, 1, MAX_CTX, self.n_state), dtype=np.float32)
        tokens = list(initial)
        logits, kv = self.dec_step([tokens], audio_features, kv, 0)
        seq = []
        sum_logprob = 0.0
        max_ts = TS_BEGIN  # monotonicity floor
        rng = np.random.default_rng(0)
        for step in range(max_tokens):
            lg = logits[0, -1].copy()
            lg[self.suppress] = -np.inf
            lg[NO_TIMESTAMPS] = -np.inf
            lg[SOT] = -np.inf
            # whisper ApplyTimestampRules semantics
            last_was_ts = len(seq) >= 1 and seq[-1] >= TS_BEGIN
            penult_was_ts = len(seq) < 2 or seq[-2] >= TS_BEGIN
            if step == 0:
                lg[:TS_BEGIN] = -np.inf  # first token must be a timestamp
            elif last_was_ts:
                if penult_was_ts:
                    lg[TS_BEGIN:] = -np.inf          # segment opened: emit text
                else:
                    lg[:EOT] = -np.inf               # segment closed: ts or EOT
                    lg[EOT + 1:TS_BEGIN] = -np.inf
            lg[TS_BEGIN:max_ts] = -np.inf            # non-decreasing timestamps
            # whisper rule: prefer timestamp if total ts prob mass beats best text
            probs = lg - lg.max()
            probs = np.exp(probs); probs /= probs.sum()
            ts_mass = probs[TS_BEGIN:].sum()
            best_text = probs[:TS_BEGIN].max() if np.isfinite(lg[:TS_BEGIN]).any() else 0
            if ts_mass > best_text and np.isfinite(lg[TS_BEGIN:]).any():
                nxt = TS_BEGIN + int(np.argmax(lg[TS_BEGIN:]))
            elif temperature == 0:
                nxt = int(np.argmax(lg))
            else:
                p = np.exp((lg - lg.max()) / temperature)
                p[~np.isfinite(p)] = 0
                p /= p.sum()
                nxt = int(rng.choice(len(p), p=p))
            # accumulate logprob of the chosen token (over unmasked dist)
            sum_logprob += float(np.log(probs[nxt] + 1e-12))
            if nxt == EOT:
                break
            seq.append(nxt)
            if nxt >= TS_BEGIN:
                max_ts = max(max_ts, nxt)
            logits, kv = self.dec_step([[nxt]], audio_features, kv, len(initial) + len(seq) - 1)
        avg_logprob = sum_logprob / max(1, len(seq) + 1)
        return seq, avg_logprob

    def transcribe(self, audio, lang="id", log=print, on_segment=None):
        segments = []
        seek = 0  # samples
        n = len(audio)
        while seek < n:
            chunk = audio[seek:seek + N_AUDIO_PER_WINDOW]
            chunk_dur = len(chunk) / SAMPLE_RATE
            if len(chunk) < N_AUDIO_PER_WINDOW:
                chunk = np.pad(chunk, (0, N_AUDIO_PER_WINDOW - len(chunk)))
            mel = log_mel(chunk)[:, :N_FRAMES]
            af = self.encode(mel)
            # whisper-style temperature fallback on low confidence / degenerate output
            import zlib
            seq = []
            for temp in (0.0, 0.2, 0.4, 0.6, 0.8):
                seq, avg_lp = self.decode_window(af, lang, temperature=temp)
                text_tokens_all = [t for t in seq if t < EOT]
                txt = self.tok.decode(text_tokens_all)
                comp_ratio = (len(txt.encode()) / max(1, len(zlib.compress(txt.encode())))) if txt else 0.0
                degenerate = comp_ratio > 2.4 or avg_lp < -1.0
                if len(text_tokens_all) > 30 and len(set(text_tokens_all[-30:])) <= 4:
                    degenerate = True
                if not degenerate:
                    break
            # parse <|t0|> text <|t1|> pairs
            base_t = seek / SAMPLE_RATE
            i = 0
            window_segments = []
            last_ts_val = 0.0
            while i < len(seq):
                if seq[i] >= TS_BEGIN:
                    t0 = (seq[i] - TS_BEGIN) * 0.02
                    j = i + 1
                    text_toks = []
                    while j < len(seq) and seq[j] < EOT:
                        text_toks.append(seq[j]); j += 1
                    t1 = (seq[j] - TS_BEGIN) * 0.02 if j < len(seq) and seq[j] >= TS_BEGIN else chunk_dur
                    if text_toks:
                        txt = self.tok.decode(text_toks).strip()
                        if txt:
                            window_segments.append((base_t + t0, base_t + min(t1, chunk_dur), txt))
                    last_ts_val = max(last_ts_val, t1)
                    i = j + 1
                else:
                    i += 1
            segments.extend(window_segments)
            for s in window_segments:
                log(f"  [{s[0]:7.1f}s] {s[2][:100]}")
                if on_segment:
                    on_segment(s)
            # advance: by last timestamp if meaningful, else full window
            if last_ts_val > 1.0 and seek + N_AUDIO_PER_WINDOW < n:
                seek += int(last_ts_val * SAMPLE_RATE)
            else:
                seek += N_AUDIO_PER_WINDOW
        return segments


def load_audio(fp, start=None, dur=None):
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ff]
    if start is not None: cmd += ["-ss", str(start)]
    if dur is not None: cmd += ["-t", str(dur)]
    cmd += ["-i", str(fp), "-f", "s16le", "-ar", "16000", "-ac", "1", "-v", "quiet", "pipe:1"]
    r = subprocess.run(cmd, capture_output=True)
    return np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def fmt(t):
    t = int(t); h, r = divmod(t, 3600); m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("audio"); ap.add_argument("out")
    ap.add_argument("--lang", default="id")
    ap.add_argument("--model", default="small", choices=["small", "medium"])
    ap.add_argument("--enc"); ap.add_argument("--dec")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--start", type=float); ap.add_argument("--dur", type=float)
    a = ap.parse_args()

    ap2 = a
    resume_sec = 0.0
    if getattr(a, "resume", False) and Path(a.out).exists():
        import re as _re
        txt = Path(a.out).read_text(encoding="utf-8")
        ts = _re.findall(r"\((?:(\d+):)?(\d+):(\d+) - (?:(\d+):)?(\d+):(\d+)\)", txt)
        if ts:
            h, m, s2 = ts[-1][3] or "0", ts[-1][4], ts[-1][5]
            resume_sec = int(h)*3600 + int(m)*60 + int(s2)
            print(f"resuming from {resume_sec:.0f}s", flush=True)
    audio = load_audio(a.audio, a.start, a.dur)
    print(f"audio: {len(audio)/16000:.0f}s; loading model...", flush=True)
    if a.model == "medium":
        model = WhisperONNX(a.enc or S / "encoder_medium.onnx", a.dec or S / "decoder_medium_fix_kv_cache.onnx", kv_layers=48, n_state=1024)
    else:
        model = WhisperONNX(a.enc or S / "encoder_small.onnx", a.dec or S / "decoder_small_fix_kv_cache.onnx")
    t0 = time.time()
    mode = "a" if resume_sec > 0 else "w"
    f = open(a.out, mode, encoding="utf-8")
    if mode == "w":
        f.write(f"Transcript: {Path(a.audio).name}\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} (whisper-{a.model} multilingual, ONNX)\n")
        f.write("=" * 60 + "\n\n")
        f.flush()
    def emit(s, _f=f, _off=resume_sec):
        t0s, t1s, txt = s
        _f.write(f"({fmt(t0s+_off)} - {fmt(t1s+_off)})\n{txt}\n\n"); _f.flush()
    segs = model.transcribe(audio[int(resume_sec*16000):], lang=a.lang, on_segment=emit)
    f.close()
    el = time.time() - t0
    print(f"done in {el:.1f}s, {len(segs)} segments -> {a.out}")

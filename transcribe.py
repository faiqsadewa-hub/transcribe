"""
Indo Transcriber - Indonesian Interview Transcription (Transcription Only)

Uses Faster-Whisper for transcription. Runs fully locally on GPU (RTX 3050 Ti compatible).
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from faster_whisper import WhisperModel

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Keep references to streaming generators to avoid premature native cleanup crashes.
_TRANSCRIBE_GC_GUARD = []


def _get_ffmpeg_path() -> str:
    """Get ffmpeg binary path, preferring imageio-ffmpeg's bundled binary."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    import shutil
    path = shutil.which("ffmpeg")
    if path:
        return path
    raise RuntimeError(
        "FFmpeg not found. Install it via: pip install imageio-ffmpeg\n"
        "Or install FFmpeg system-wide: https://ffmpeg.org/download.html"
    )


# ─── Utility Functions ────────────────────────────────────────────────────────

def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def safe_console_text(text: str) -> str:
    """Return text safe to print in the current terminal encoding."""
    encoding = sys.stdout.encoding or "utf-8"
    try:
        text.encode(encoding)
        return text
    except UnicodeEncodeError:
        return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def load_audio(file_path: str) -> tuple[np.ndarray, int]:
    """
    Load audio file and convert to 16kHz mono float32 numpy array.
    Supports .m4a, .mp3, .wav, .flac, etc. via ffmpeg.
    """
    import subprocess

    print(f"  Loading audio: {Path(file_path).name}")

    ffmpeg = _get_ffmpeg_path()

    cmd = [
        ffmpeg, "-i", str(file_path),
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-v", "quiet",
        "pipe:1",
    ]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed to decode audio: {result.stderr.decode(errors='replace')}"
        )

    audio_np = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32)
    audio_np /= 32768.0
    sample_rate = 16000

    duration = len(audio_np) / sample_rate
    print(f"  Audio loaded: {format_timestamp(duration)} duration, {sample_rate}Hz mono")
    return audio_np, sample_rate


# ─── Transcription ────────────────────────────────────────────────────────────

def load_whisper_model(model_size: str, device: str) -> WhisperModel:
    """Load Whisper model once and reuse it for all files."""
    print(f"\n[Model] Loading Whisper '{model_size}' on {device}...")
    compute_type = "float16" if device == "cuda" else "int8"
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe_audio(audio: np.ndarray, model: WhisperModel, initial_prompt: str | None = None) -> list:
    """
    Transcribe audio using Faster-Whisper.
    Returns list of segments with start, end, text.
    """
    print("  [Transcription] Transcribing (language=Indonesian)...")
    if initial_prompt:
        print(f"  [Transcription] Using initial prompt ({len(initial_prompt)} chars)")

    segments_gen, info = model.transcribe(
        audio,
        language="id",
        beam_size=5,
        temperature=0,
        condition_on_previous_text=False,
        initial_prompt=initial_prompt,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=300,
            speech_pad_ms=300,
        ),
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.0,
        word_timestamps=True,
    )
    print(f"  [Transcription] Detected language: {info.language} (prob={info.language_probability:.2f})")
    _TRANSCRIBE_GC_GUARD.append(segments_gen)

    segments = []
    for seg in segments_gen:
        text = seg.text.strip()
        segments.append({
            "start": seg.start,
            "end": seg.end,
            "text": text,
        })
        ts = format_timestamp(seg.start)
        preview = text[:80] + "..." if len(text) > 80 else text
        print(f"  [{ts}] {safe_console_text(preview)}", flush=True)

    print(f"\n  [Transcription] Done - {len(segments)} segments found")
    sys.stdout.flush()

    return segments


# ─── Output ───────────────────────────────────────────────────────────────────

def write_transcript(segments: list, output_path: str | Path, source_file: str) -> str:
    """Write transcript to .txt file with timestamps and return absolute path."""
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        f.write(f"Transcript: {Path(source_file).name}\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        if not segments:
            f.write("(No speech detected)\n")
        else:
            for seg in segments:
                ts_start = format_timestamp(seg["start"])
                ts_end = format_timestamp(seg["end"])
                f.write(f"({ts_start} - {ts_end})\n")
                f.write(f"{seg['text']}\n\n")

    if not output_path.exists():
        raise RuntimeError(f"Transcript file was not created: {output_path}")

    print(f"  [Output] Saved to: {output_path}")
    return str(output_path)


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def process_file(file_path: str, output_dir: Path, model: WhisperModel, initial_prompt: str | None = None) -> str:
    """Full pipeline: load → transcribe → save."""
    source_path = Path(file_path)
    file_name = source_path.stem
    output_path = output_dir / f"{file_name}.txt"

    print(f"\n{'='*60}")
    print(f"Processing: {source_path.name}")
    print(f"{'='*60}")

    start_time = time.time()

    audio, _ = load_audio(file_path)
    segments = transcribe_audio(audio, model=model, initial_prompt=initial_prompt)

    if not segments:
        print("  [WARNING] No speech detected in audio file. Writing empty transcript.")

    print("  [Output] Writing transcript file...")
    saved_path = write_transcript(segments, output_path, file_path)

    elapsed = time.time() - start_time
    print(f"\n  Completed in {format_timestamp(elapsed)}")
    return saved_path


def main():
    parser = argparse.ArgumentParser(
        description="Indonesian Interview Transcriber",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transcribe.py interview.m4a
  python transcribe.py ./audio_folder/
  python transcribe.py interview.m4a --output ./transcripts/
  python transcribe.py interview.m4a --model medium
  python transcribe.py interview.m4a --model large-v3 --prompt "Rapat kickoff proyek BCAS"
        """,
    )
    parser.add_argument(
        "input",
        help="Path to audio file or folder containing audio files",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory for transcripts (default: same as input)",
    )
    parser.add_argument(
        "--model", "-m",
        default="small",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size (default: small). Larger = more accurate but slower & more VRAM.",
    )
    parser.add_argument(
        "--prompt", "-p",
        default=None,
        help="Initial prompt to guide transcription with domain vocabulary and context.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU mode (slower, but no GPU required).",
    )

    args = parser.parse_args()

    # ── Resolve device ───────────────────────────────────────────
    if args.cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name} ({vram:.1f} GB VRAM)")
    else:
        print("[WARNING] CUDA not available - falling back to CPU (this will be slow)")
        device = "cpu"

    # ── Resolve input files ──────────────────────────────────────
    input_path = Path(args.input).resolve()
    if input_path.is_file():
        files = [str(input_path)]
        default_output = str(input_path.parent)
    elif input_path.is_dir():
        extensions = {".m4a", ".mp3", ".wav", ".flac"}
        files = sorted(
            str(f) for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in extensions
        )
        if not files:
            print(f"[ERROR] No audio files found in: {input_path}")
            sys.exit(1)
        default_output = str(input_path)
    else:
        print(f"[ERROR] Input not found: {input_path}")
        sys.exit(1)

    # ── Resolve output directory ─────────────────────────────────
    output_dir = Path(args.output).expanduser().resolve() if args.output else Path(default_output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Process files ────────────────────────────────────────────
    print(f"\nFiles to process: {len(files)}")
    print(f"Output directory: {output_dir}")
    print(f"Model: whisper-{args.model} | Device: {device}")
    if args.prompt:
        print(f"Initial prompt: {args.prompt[:100]}...")

    model = load_whisper_model(model_size=args.model, device=device)

    total_start = time.time()
    saved_paths = []
    failed_files = []

    for i, file_path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}]", end="")
        try:
            saved_path = process_file(
                file_path=file_path,
                output_dir=output_dir,
                model=model,
                initial_prompt=args.prompt,
            )
            saved_paths.append(saved_path)
        except Exception as e:
            failed_name = Path(file_path).name
            failed_files.append(failed_name)
            print(f"\n  [ERROR] Failed to process {failed_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(
        f"All done! Success: {len(saved_paths)}/{len(files)} file(s) in {format_timestamp(total_elapsed)}"
    )
    print(f"Transcripts saved to: {output_dir}")
    if saved_paths:
        print("Saved transcript files:")
        for path in saved_paths:
            print(f"  - {path}")
    if failed_files:
        print("Failed files:")
        for name in failed_files:
            print(f"  - {name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

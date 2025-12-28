import os
import tempfile
import time
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any

import runpod

from libs.bbcore.s3 import from_env as s3_from_env, S3
from libs.bbcore.logging import *  # noqa: F401,F403

# Expected input payload
# {
#   "projectId": "...",
#   "takeId": "...",
#   "jobId": "...",
#   "audioUrl": "https://..." | null,
#   "audioKey": "projects/.../input.wav" | null,
#   "mode": "vocals" | "4stem" | "6stem",
#   "quality": "fast" | "balanced" | "best"
# }


def download_input(s3: S3, payload: Dict[str, Any], dst: Path) -> None:
    url = payload.get("audioUrl")
    key = payload.get("audioKey")
    if key:
        s3.download_file_to_path(key, str(dst))
        return
    if url:
        import requests
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        with open(dst, "wb") as f:
            shutil.copyfileobj(r.raw, f)
        return
    raise ValueError("audioUrl or audioKey is required")


def run_demucs(input_path: Path, out_dir: Path, mode: str, quality: str) -> Path:
    cmd = [
        "demucs",
        "--out",
        str(out_dir),
    ]
    if mode == "vocals":
        cmd += ["--two-stems", "vocals"]
    elif mode == "6stem":
        cmd += ["--name", "htdemucs_6s"]  # placeholder; switch to 6-stem model if available
    # quality tiers could map to segment or model variant
    if quality == "fast":
        cmd += ["--segment", "8"]
    elif quality == "best":
        cmd += ["--name", "htdemucs_ft"]
    cmd.append(str(input_path))

    subprocess.run(cmd, check=True)

    # demucs writes under out_dir/<model_name>/<basename>
    # Find first directory with stems
    candidates = list(out_dir.glob("**/*"))
    for c in candidates:
        if c.is_dir() and any((c / x).exists() for x in ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]):
            return c
    # Fallback: search widely
    for c in out_dir.glob("**/*.wav"):
        pass
    raise RuntimeError("Demucs output not found")


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    start = time.time()
    project_id = event["projectId"]
    take_id = event["takeId"]
    job_id = event["jobId"]
    mode = event.get("mode", "4stem")
    quality = event.get("quality", "balanced")

    s3 = s3_from_env()
    s3.ensure_bucket()

    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        input_path = tmpdir / "input.audio"
        download_input(s3, event, input_path)

        out_dir = tmpdir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        stems_dir = run_demucs(input_path, out_dir, mode, quality)

        rel_base = f"analysis/separation/{job_id}/"
        keys = {}
        def upload(name: str, filename: str):
            p = stems_dir / filename
            if p.exists():
                key = S3.build_key(project_id, take_id, rel_base + filename)
                url = s3.upload_file(str(p), key, content_type="audio/wav")
                keys[name] = s3.presign(key)

        upload("vocals", "vocals.wav")
        upload("drums", "drums.wav")
        upload("bass", "bass.wav")
        upload("other", "other.wav")
        upload("piano", "piano.wav")
        upload("guitar", "guitar.wav")

        duration = None
        sample_rate = None
        channels = None
        try:
            import soundfile as sf
            with sf.SoundFile(stems_dir / "vocals.wav") as f:
                duration = len(f) / f.samplerate
                sample_rate = f.samplerate
                channels = f.channels
        except Exception:
            pass

        return {
            "jobId": job_id,
            "stems": keys,
            "processingTime": time.time() - start,
            "metadata": {
                "duration": duration or 0,
                "sampleRate": sample_rate or 48000,
                "channels": channels or 2,
            },
        }


runpod.serverless.start({"handler": handler})

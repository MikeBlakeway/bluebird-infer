import os
import tempfile
import time
import shutil
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

import runpod
from opentelemetry import trace, metrics
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

from libs.bbcore.s3 import from_env as s3_from_env, S3
from libs.bbcore.logging import *  # noqa: F401,F403

logger = logging.getLogger(__name__)

# Initialize OTEL tracing
_otel_endpoint = os.getenv("BB_OTEL_ENDPOINT", "http://localhost:4317")
_tracer = None

def init_otel():
  """Initialize OpenTelemetry tracing."""
  global _tracer
  try:
    resource = Resource.create({"service.name": "separation-pod"})
    otlp_exporter = OTLPSpanExporter(endpoint=_otel_endpoint)
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
    trace.set_tracer_provider(tracer_provider)
    _tracer = trace.get_tracer(__name__)
    RequestsInstrumentor().instrument()
    logger.info(f"OTEL initialized: endpoint={_otel_endpoint}")
  except Exception as e:
    logger.warning(f"Failed to initialize OTEL: {e}")
    _tracer = None

init_otel()


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

@dataclass
class DemucsConfig:
  """Config for Demucs model and processing."""
  model: str
  segment: Optional[int]  # chunk size in seconds; None = no chunking

def get_demucs_config(mode: str, quality: str) -> DemucsConfig:
  """Map output mode and quality to Demucs model and segment settings."""
  # Quality mapping: quality affects segment size (faster = smaller chunks)
  # fast: 8s chunks (quicker but may have edge artifacts)
  # balanced: default segment (best quality/speed tradeoff)
  # best: 32s chunks (slower but highest quality)
  quality_segments = {
    "fast": 8,
    "balanced": None,  # Use Demucs default (~11s)
    "best": 32,
  }
  segment = quality_segments.get(quality, None)

  # Mode mapping: select appropriate Demucs output
  if mode == "vocals":
    # 2-stem: vocals + accompaniment
    return DemucsConfig(model="htdemucs", segment=segment)
  elif mode == "4stem":
    # 4-stem: vocals, drums, bass, other (Demucs default)
    return DemucsConfig(model="htdemucs", segment=segment)
  elif mode == "6stem":
    # 6-stem: vocals, drums, bass, other, piano, guitar
    # Use htdemucs_6s if available, else fall back to default
    return DemucsConfig(model="htdemucs_6s", segment=segment)
  else:
    raise ValueError(f"Unknown mode: {mode}")


def download_input(s3: S3, payload: Dict[str, Any], dst: Path) -> None:
  """Download input audio from S3 key or URL."""
  url = payload.get("audioUrl")
  key = payload.get("audioKey")
  if key:
    logger.info(f"Downloading from S3 key: {key}")
    s3.download_file_to_path(key, str(dst))
    return
  if url:
    logger.info(f"Downloading from URL: {url}")
    import requests
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(dst, "wb") as f:
      shutil.copyfileobj(r.raw, f)
    return
  raise ValueError("audioUrl or audioKey is required")



def run_demucs(input_path: Path, out_dir: Path, config: DemucsConfig) -> Path:
  """Run Demucs CLI with config, return path to stems directory."""
  logger.info(f"Running Demucs model={config.model} segment={config.segment}")
  cmd = [
    "demucs",
    "--out", str(out_dir),
    "--name", config.model,
  ]
  if config.segment:
    cmd += ["--segment", str(config.segment)]
  cmd.append(str(input_path))

  logger.info(f"Demucs command: {' '.join(cmd)}")
  subprocess.run(cmd, check=True)

  # demucs writes under out_dir/<model_name>/<basename>
  # Find first directory with stems
  for candidate in out_dir.glob(f"{config.model}/*"):
    if candidate.is_dir() and any((candidate / x).exists() for x in ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]):
      logger.info(f"Found stems at: {candidate}")
      return candidate
  raise RuntimeError(f"Demucs output not found in {out_dir}")



def collect_stems(stems_dir: Path, mode: str) -> Dict[str, Path]:
  """Collect available stems based on output mode."""
  stems = {}
  if mode == "vocals":
    # 2-stem mode: vocals and accompaniment
    if (stems_dir / "vocals.wav").exists():
      stems["vocals"] = stems_dir / "vocals.wav"
    if (stems_dir / "accompaniment.wav").exists():
      stems["accompaniment"] = stems_dir / "accompaniment.wav"
    elif (stems_dir / "other.wav").exists():
      # Fallback if accompaniment not split out
      stems["accompaniment"] = stems_dir / "other.wav"
  elif mode in ("4stem", "6stem"):
    # Multi-stem modes: collect all available
    for name in ["vocals", "drums", "bass", "other", "piano", "guitar"]:
      path = stems_dir / f"{name}.wav"
      if path.exists():
        stems[name] = path
  return stems

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
  """Main Demucs separation handler."""
  start = time.time()
  project_id = event["projectId"]
  take_id = event["takeId"]
  job_id = event["jobId"]
  mode = event.get("mode", "4stem")
  quality = event.get("quality", "balanced")

  # Create OTEL span
  if _tracer:
    with _tracer.start_as_current_span("separation_job") as span:
      span.set_attribute("job_id", job_id)
      span.set_attribute("project_id", project_id)
      span.set_attribute("take_id", take_id)
      span.set_attribute("mode", mode)
      span.set_attribute("quality", quality)
      return _run_separation(project_id, take_id, job_id, mode, quality, start)
  else:
    return _run_separation(project_id, take_id, job_id, mode, quality, start)

def _run_separation(project_id: str, take_id: str, job_id: str, mode: str, quality: str, start: float) -> Dict[str, Any]:
  """Core separation logic."""
  logger.info(f"Starting separation job {job_id}: mode={mode}, quality={quality}")

  s3 = s3_from_env()
  s3.ensure_bucket()

  with tempfile.TemporaryDirectory() as td:
    tmpdir = Path(td)
    input_path = tmpdir / "input.audio"
    download_input(s3, {"audioUrl": None, "audioKey": None, **{"audioUrl": None, "audioKey": None}}, input_path)
    logger.info(f"Downloaded input to {input_path}")

    # Get Demucs config based on mode and quality
    config = get_demucs_config(mode, quality)

    out_dir = tmpdir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if _tracer:
      with _tracer.start_as_current_span("demucs_run") as span:
        span.set_attribute("model", config.model)
        span.set_attribute("segment", config.segment or 0)
        stems_dir = run_demucs(input_path, out_dir, config)
    else:
      stems_dir = run_demucs(input_path, out_dir, config)
    
    logger.info(f"Demucs completed in {time.time() - start:.2f}s")

    # Collect stems based on output mode
    stems_paths = collect_stems(stems_dir, mode)
    logger.info(f"Collected {len(stems_paths)} stems: {list(stems_paths.keys())}")

    # Upload stems to S3
    rel_base = f"analysis/separation/{job_id}/"
    keys = {}
    if _tracer:
      with _tracer.start_as_current_span("s3_upload") as span:
        span.set_attribute("stem_count", len(stems_paths))
        for name, path in stems_paths.items():
          key = S3.build_key(project_id, take_id, rel_base + f"{name}.wav")
          url = s3.upload_file(str(path), key, content_type="audio/wav")
          keys[name] = s3.presign(key)
          logger.info(f"Uploaded {name} to S3: {key}")
    else:
      for name, path in stems_paths.items():
        key = S3.build_key(project_id, take_id, rel_base + f"{name}.wav")
        url = s3.upload_file(str(path), key, content_type="audio/wav")
        keys[name] = s3.presign(key)
        logger.info(f"Uploaded {name} to S3: {key}")

    # Extract metadata from first available stem
    duration = None
    sample_rate = None
    channels = None
    first_stem = next(iter(stems_paths.values()), None)
    if first_stem:
      try:
        import soundfile as sf
        with sf.SoundFile(first_stem) as f:
          duration = len(f) / f.samplerate
          sample_rate = f.samplerate
          channels = f.channels
        logger.info(f"Metadata: duration={duration:.2f}s, sr={sample_rate}, channels={channels}")
      except Exception as e:
        logger.warning(f"Failed to extract metadata: {e}")

    processing_time = time.time() - start
    result = {
      "jobId": job_id,
      "stems": keys,
      "processingTime": processing_time,
      "metadata": {
        "duration": duration or 0,
        "sampleRate": sample_rate or 48000,
        "channels": channels or 2,
      },
    }
    logger.info(f"Separation job {job_id} complete in {processing_time:.2f}s")
    return result




runpod.serverless.start({"handler": handler})

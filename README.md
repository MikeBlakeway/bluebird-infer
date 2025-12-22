# Bluebird Inference

Python inference pods for the Bluebird AI music platform.

## Overview

This repository contains the Python-based inference services (pods) that power Bluebird's audio generation and analysis capabilities:

- **Analyzer Pod**: Feature extraction (key, BPM, contour, n-grams) using librosa
- **Music Pod**: Procedural music synthesis (drums, bass, guitar)
- **Melody Pod**: Procedural melody generation with chord progression awareness
- **Voice Pod**: Singing voice synthesis using DiffSinger
- **Similarity Pod**: Melody and rhythm similarity checking (n-gram Jaccard, DTW)

## Architecture

```bash
bluebird-infer/
├── libs/              # Shared libraries
│   ├── bbcore/        # Core utilities (S3, audio I/O, config, logging)
│   ├── bbfeatures/    # Feature extraction utilities
│   └── bbmelody/      # Melody generation utilities
├── pods/              # FastAPI services
│   ├── analyzer/
│   ├── music/
│   ├── melody/
│   ├── voice/
│   └── similarity/
└── tests/             # Pytest test suite
```

## Setup

### Prerequisites

- Python 3.8+
- Poetry 1.7+
- Docker & Docker Compose
- FFmpeg

### Local Development

```bash
# Install dependencies
poetry install

# Run tests
poetry run pytest

# Format code
poetry run black .
poetry run ruff check .

# Type checking
poetry run mypy libs/
```

docker-compose up analyzer
docker-compose up --build
### Docker Compose

Use the pod compose file to run analyzer and music locally (expects an external `bluebird-network`; create it with `docker network create bluebird-network` if needed and ensure MinIO/Redis are reachable there).

```bash
# Start analyzer + music pods
docker compose -f docker-compose.pods.yml up --build analyzer music

# Start a single pod
docker compose -f docker-compose.pods.yml up --build analyzer

# Tear down
docker compose -f docker-compose.pods.yml down
```

## Pod Endpoints

| Pod        | Port | Endpoint              | Purpose                   |
| ---------- | ---- | --------------------- | ------------------------- |
| Analyzer   | 8001 | POST /extract         | Extract audio features    |
| Music      | 8002 | POST /synthesize      | Generate music stems      |
| Melody     | 8003 | POST /generate-melody | Generate melodic contours |
| Voice      | 8004 | POST /synthesize      | Generate singing vocals   |
| Similarity | 8005 | POST /check           | Check melody similarity   |

## Environment Variables

Each pod requires:

- `BB_S3_ENDPOINT`: MinIO/S3 endpoint (e.g., `http://localhost:9000`)
- `BB_S3_ACCESS_KEY`: S3 access key
- `BB_S3_SECRET`: S3 secret key
- `BB_BUCKET`: S3 bucket name (default: `bluebird`)
- `BB_OTEL_ENDPOINT`: OpenTelemetry collector endpoint (optional)
- `BB_LOG_LEVEL`: Logging level (default: `INFO`)

## Testing

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=libs --cov-report=html

# Run specific test
poetry run pytest tests/test_bbcore/test_audio.py
```

## Integration with Bluebird API

The main Bluebird Node.js API (in the `bluebird` repository) communicates with these pods via HTTP. Workers in the API enqueue jobs, then call the appropriate pod endpoint with S3 URLs for input/output.

**Workflow:**

1. API worker fetches input from S3 (or receives request payload)
2. API calls pod endpoint (e.g., `POST http://analyzer:8001/extract`)
3. Pod downloads audio from S3, processes it, uploads result to S3
4. Pod returns S3 URL or result JSON to API
5. API worker updates database and emits SSE progress event

## Development Guidelines

- **Type Safety**: Use type hints throughout; run `mypy` before committing
- **Testing**: Maintain ≥70% test coverage; use golden fixtures for audio tests
- **Determinism**: All inference must accept a `seed` parameter for reproducibility
- **Performance**: Target <5s for feature extraction, <2s for music synthesis, <8s for voice synthesis
- **Error Handling**: Return structured errors with status codes (400/422/500)
- **Logging**: Use structured JSON logs with correlation IDs

## License

Apache 2.0

## Related Repositories

- **bluebird**: Main Node.js/TypeScript API and web frontend
- **DiffSinger**: OpenVPI fork for singing voice synthesis (external dependency)

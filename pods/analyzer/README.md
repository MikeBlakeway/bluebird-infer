# Analyzer Pod

Feature extraction service for Bluebird. Exposes key detection, BPM detection, and similarity checking via HTTP.

## Endpoints

### Health

- `POST /health` - Health check

### Analysis

- `POST /analyze/key` - Detect musical key from audio
- `POST /analyze/bpm` - Detect tempo (BPM) from audio
- `POST /analyze` - Full audio analysis (key + BPM)

### Similarity

- `POST /similarity/ngram` - N-gram Jaccard similarity between melodies

## Usage

### Local Development

```bash
cd pods/analyzer
poetry run python -m uvicorn main:app --reload --port 8001
```

### Docker

```bash
# Build
docker build -f Dockerfile.analyzer -t bluebird-analyzer .

# Run
docker run -p 8001:8001 bluebird-analyzer
```

### API Example

```bash
# Analyze audio file
curl -X POST http://localhost:8001/analyze \
  -F "file=@song.wav"

# Detect key only
curl -X POST http://localhost:8001/analyze/key \
  -F "file=@song.wav"

# Check similarity
curl -X POST http://localhost:8001/similarity/ngram \
  -F "reference_file=@reference.wav" \
  -F "generated_file=@generated.wav"
```

## Configuration

Environment variables:

- `ANALYZER_PORT` - Port to listen on (default: 8001)
- `ENV` - Environment (dev/prod, default: dev)
- `LOG_LEVEL` - Logging level (default: DEBUG in dev, INFO in prod)
- `OTEL_ENABLED` - Enable OpenTelemetry (default: false)
- `OTEL_EXPORTER_OTLP_ENDPOINT` - OTEL endpoint (default: <http://localhost:4317>)

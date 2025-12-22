# Music Synthesis Pod

Procedural music generation for drums, bass, and guitar tracks with perfect grid alignment.

## Features

- **Deterministic Synthesis**: Same seed → identical output
- **Multi-Instrument**: Drums, bass, guitar synthesis
- **Grid Alignment**: Sample-accurate rendering at BPM boundaries
- **Multi-Stem Output**: Separate tracks for mixing
- **Performance**: <2s per section on CPU

## Architecture

### synth.py

**DrumSynth**

- Kick (sine sweep decay)
- Snare (filtered noise with envelope)
- Hi-hat (bright noise burst)
- Pattern generation (4/4 beats)

**BassSynth**

- MIDI note to frequency conversion (A4 = 440Hz)
- Multiple waveforms: sine, sawtooth, square, triangle
- ADSR envelope
- Bass line generation from MIDI sequences

**GuitarSynth**

- Karplus-Strong-like pluck synthesis
- Multi-harmonic oscillation
- Exponential decay envelope
- Chord strumming simulation

### grid.py

**GridAligner**

- Beat position to sample index conversion
- Section boundary calculation
- Audio quantization to grid
- Stem alignment

**GridRenderer**

- Section-level rendering
- Stem mixing with volume levels
- Automatic normalization
- Grid-perfect output

### main.py

FastAPI application with endpoints:

- `POST /health` - Service health check
- `POST /synthesize` - Generate music section

## API

### POST /synthesize

Generate a music section.

**Request:**

```json
{
  "bpm": 120,
  "duration": 8.0,
  "include_drums": true,
  "include_bass": true,
  "include_guitar": false,
  "seed": 42,
  "bass_note": 36,
  "master_volume": 1.0
}
```

**Response:**

```json
{
  "duration": 8.0,
  "sample_rate": 48000,
  "bit_depth": 24,
  "stems": {
    "drums": "base64-encoded-wav",
    "bass": "base64-encoded-wav"
  },
  "mixed_audio": "base64-encoded-wav",
  "message": "Generated 2 stems at 120 BPM"
}
```

## Usage

### Local

```bash
# Start pod
poetry run python -m uvicorn main:app --host 0.0.0.0 --port 8002

# Test
curl -X POST http://localhost:8002/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "bpm": 120,
    "duration": 8.0,
    "include_drums": true,
    "include_bass": true,
    "seed": 42
  }'
```

### Docker

```bash
docker build -f Dockerfile.music -t bluebird-music:latest .
docker run -p 8002:8002 bluebird-music:latest
```

## Configuration

Environment variables:

- `MUSIC_PORT` - Service port (default: 8002)
- `MUSIC_ENV` - Environment (dev/prod)
- `MUSIC_SAMPLE_RATE` - Sample rate (default: 48000)

## Performance

- Drum pattern generation: ~200ms
- Bass line generation: ~150ms
- Grid alignment: ~50ms
- Total per 8-second section: <1s

## Testing

```bash
poetry run pytest tests/test_pods/test_music.py -v
```

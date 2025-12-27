# Similarity Pod

Melody and rhythm similarity checking service for export gating.

## Overview

Checks if a generated melody is too similar to a reference melody. Returns a **verdict** (pass/borderline/block) with a **combined score** and **recommendations**.

- **Melody Similarity**: n-gram Jaccard distance (interval preservation)
- **Rhythm Similarity**: DTW distance (timing/IOI preservation)
- **Combined Score**: 0.6 × melody + 0.4 × rhythm
- **Verdicts**:
  - **pass** (score < 0.35): Safe to export
  - **borderline** (0.35 ≤ score < 0.48): Show warning, allow pro users to override
  - **block** (score ≥ 0.48): Prevent export, recommend regeneration

## API

### POST /health

Health check.

**Response:**
```json
{
  "status": "ok",
  "service": "similarity",
  "version": "0.1.0"
}
```

### POST /check

Check similarity between two melodies.

**Request:**
```json
{
  "reference_melody": [60, 62, 64, 65, 67],
  "generated_melody": [60, 62, 64, 65, 69]
}
```

**Response (Pass):**
```json
{
  "melody_score": 0.2,
  "rhythm_score": 0.15,
  "combined_score": 0.18,
  "verdict": "pass",
  "confidence": 0.95,
  "recommendations": []
}
```

**Response (Borderline):**
```json
{
  "melody_score": 0.4,
  "rhythm_score": 0.45,
  "combined_score": 0.42,
  "verdict": "borderline",
  "confidence": 0.85,
  "recommendations": [
    "Shift melody key by +1 semitone",
    "Regenerate chorus topline"
  ]
}
```

**Response (Block):**
```json
{
  "melody_score": 0.8,
  "rhythm_score": 0.85,
  "combined_score": 0.82,
  "verdict": "block",
  "confidence": 0.98,
  "recommendations": [
    "Use different reference for remix",
    "Regenerate with fresh melody seed"
  ]
}
```

## Running Locally

```bash
# Start pod alone
docker compose -f docker-compose.pods.yml up --build similarity

# Or with other pods
docker compose -f docker-compose.pods.yml up --build analyzer music melody similarity
```

## Testing

```bash
# Health check
curl -X POST http://localhost:8005/health

# Similarity check (borderline case)
curl -X POST http://localhost:8005/check \
  -H "Content-Type: application/json" \
  -d '{
    "reference_melody": [60, 62, 64, 65, 67, 69, 71, 72],
    "generated_melody": [60, 62, 64, 65, 67, 69, 71, 74]
  }'
```

## Development

- **Tests**: `tests/test_pods/test_similarity.py`
- **Golden Fixtures**: `tests/fixtures/similarity/`
- **Algorithm**: Based on interval n-grams (Jaccard) + DTW rhythm distance

## TODO (Day 2+)

- [ ] Implement n-gram Jaccard similarity
- [ ] Implement DTW rhythm comparison
- [ ] Calibrate thresholds with golden fixtures
- [ ] Generate user recommendations
- [ ] Add S3 artifact handling (optional)

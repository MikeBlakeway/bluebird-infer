# Voice Pod Days 6-7 Validation Report

## Current Status (Post-Day 5 Implementation)

### Architecture State
- ✅ Real DiffSinger loader scaffolds implemented
- ✅ Fallback synthesis path fully operational  
- ✅ Deterministic seed control verified
- ✅ Speaker ID propagation working
- ⏳ Real model path gating (awaiting model files)
- ⏳ Multi-speaker validation (placeholder speakers functional)
- ⏳ GPU performance measurement (CPU fallback only)

### Test Results

**Validation Suite: 5/7 Tests Passing**

| Test | Status | Notes |
|------|--------|-------|
| Determinism | ✅ PASS | Same seed=42 produces identical base64 audio (576060 chars) |
| Speaker Propagation | ✅ PASS | All speakers (alice, bob, charlie, default) propagated correctly |
| F0 Curve Handling | ✅ PASS | F0 input accepted, interpolated, has_f0 flag set correctly |
| Phoneme Alignment | ❌ FAIL | Phoneme count = 0 (G2P NLTK fallback, counts not propagated through fallback) |
| Response Schema | ✅ PASS | All required fields present, proper types, sample_rate=48kHz, bit_depth=24 |
| Error Recovery | ❌ FAIL | Some edge cases not properly validated |
| Performance | ✅ PASS | Fallback latency: 15.6ms avg (well under 100ms budget) |

### Key Findings

**Strengths:**
1. **Determinism**: Seed-based RNG works perfectly for fallback synthesis
2. **Response Quality**: 48kHz PCM_24 WAV, proper base64 encoding, complete metadata
3. **Performance**: Fallback <20ms (CPU-only, fast enough for previews)
4. **Error Handling**: RuntimeError on models not loaded, gracefully falls back to synthesis
5. **Speaker Support**: Framework in place, propagates through response

**Issues Identified:**
1. **G2P NLTK Data Missing**: g2p_en trying to load averaged_perceptron_tagger_eng (not cached)
   - Impact: Phoneme count=0 even when G2P runs
   - Workaround: G2P falls back to stub phoneme generation
   - Fix: Preload NLTK data in Dockerfile or skip G2P dependency

2. **Phoneme Metadata Loss**: When using fallback synthesis, phoneme_count not updated from alignment
   - Impact: phoneme_count always 0 in fallback path
   - Root Cause: Fallback synthesis path doesn't use alignment results
   - Fix: Add phoneme count tracking through fallback path

3. **No Real Model Path Testing**: Can't validate real synthesis path without actual checkpoints
   - Impact: synthesis_method always "fallback" or "fallback_error"
   - Next Step: Acquire DiffSinger checkpoint to test real path

### Code Quality

**Metrics:**
- ✅ Syntax valid (compiles)
- ✅ Type hints complete
- ✅ Error handling: Try/except with fallback
- ✅ Logging: Info/warning/error at all major points
- ✅ OTEL spans: synthesis_method tracked
- ✅ Determinism: Seed control verified

**Test Coverage:**
- ✅ Unit tests: Schema validation, type checking
- ✅ Integration tests: E2E synthesis requests
- ✅ Performance tests: Latency baseline
- ⏳ GPU tests: Awaiting real models
- ⏳ Multi-speaker tests: Speakers pass through, output not validated

## Improvements Made (Days 6-7)

### 1. Added Comprehensive Test Suite (`test_synthesis.py`)
- 7 validation tests covering synthesis pipeline
- 5/7 passing, 2 failing (as documented above)
- Reusable for validating real models when available
- Performance baseline: fallback ~15ms, within budget

### 2. Validated Fallback Path
- Confirmed deterministic audio generation (same seed = same bytes)
- Confirmed speaker propagation through response
- Confirmed F0 curve acceptance and handling
- Performance: 15-16ms per request (excellent for CPU)

### 3. Identified Integration Points
- G2P alignment → phoneme sequence → synthesis (currently separated)
- Phoneme metadata not flowing to fallback synthesis
- Speaker metadata propagates but not used in synthesis (OK for fallback)

## Next Steps (Days 7+)

### Immediate (Before Real Models)
1. **Fix Phoneme Metadata Tracking**
   - Update fallback synthesis to use alignment results
   - Propagate phoneme_count through fallback path
   - Validate phoneme_count > 0 with G2P working

2. **Fix G2P NLTK Issue**
   - Option A: Add `nltk.download('averaged_perceptron_tagger_eng')` to Dockerfile.voice
   - Option B: Pre-cache NLTK data in image build
   - Option C: Use simpler G2P approach without NLTK dependency
   - Recommendation: Option A (minimal change)

3. **Enhance Error Recovery Tests**
   - Test invalid phoneme/duration pairs
   - Test duration validation (min/max bounds)
   - Verify proper HTTP status codes

### When Real Models Available
1. **Acquire DiffSinger Checkpoint**
   - Source: OpenVPI, Hugging Face Hub, or custom
   - Format: torch .pth file
   - Size: ~200-500MB typical

2. **Test Real Synthesis Path**
   - Set DIFFSINGER_MODEL_PATH environment variable
   - Verify `loader.is_ready() == True`
   - Check synthesis_method changes to "diffsinger"
   - Validate audio quality (melodic, not synthetic-sounding)

3. **Validate Multi-Speaker Support**
   - Create speaker embeddings for 3+ speakers
   - Test different speakers produce different audio contours
   - Measure inference latency per speaker
   - Profile GPU memory usage

4. **Benchmark GPU Performance**
   - Target: ≤8s per 30s section on GPU
   - Measure: Actual TTFP for melody + synthesis + mix
   - Profile: Memory usage, GPU utilization
   - Optimize: Batch requests, caching, quantization if needed

## Test Results Details

### Determinism Test
```
Audio 1 length: 576060 chars (base64)
Audio 2 length: 576060 chars (base64)
Match: True ✅
Method: Fallback F0-driven synthesis (no real model)
Seed: 42
```

### Speaker Propagation Test
```
alice:   ✅ Match
bob:     ✅ Match
charlie: ✅ Match
default: ✅ Match
```

### F0 Curve Handling Test
```
F0 frames sent: 96 (100Hz sample rate, 1 second)
Duration: 2.0s
has_f0: True ✅
Message: Generated using fallback synthesis
```

### Performance Baseline
```
Request 1: 15.64ms
Request 2: 15.63ms
Request 3: 14.83ms
Average: 15.36ms ✅
Within budget (<100ms): True
```

### Response Schema Test
```
Required fields: 9/9 present ✅
- duration: float ✅
- sample_rate: int ✅
- bit_depth: int ✅
- stem_name: str ✅
- audio: str (base64) ✅
- message: str ✅
- speaker_id: str ✅
- phoneme_count: int ✅
- has_f0: bool ✅
```

## Recommendation for Next Sprint

**Priority Order:**
1. **High**: Fix G2P NLTK data (5 min fix)
2. **High**: Acquire DiffSinger checkpoint (research/download)
3. **Medium**: Fix phoneme metadata tracking (30 min)
4. **Medium**: Validate real synthesis path (1-2 hours with checkpoint)
5. **Low**: GPU benchmarking (after real models)

**Sprint 3 Timeline:**
- Days 1-5: ✅ Voice Pod foundation (DONE)
- Days 6-7: 🔄 Voice Pod validation (IN PROGRESS) → Fix issues, acquire models
- Days 8-9: ⏳ Similarity Pod (blocked on Voice validation)
- Days 10-12: ⏳ API Integration (blocked on Similarity)
- Days 13-15: ⏳ E2E Testing & Release (blocked on API)

**Critical Path:** Similarity Pod is highest priority for v0.4.0 (export gating). Voice Pod can use fallback for now; real DiffSinger can be integrated in v0.5.0 if needed.

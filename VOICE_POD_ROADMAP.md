# Voice Pod Implementation - Remaining Work (Days 6+)

## Sprint Context
- **Current Sprint**: Sprint 3 (Real Model Integration)
- **Previous Sprints**: Days 1-4 (G2P + Model scaffolds complete)
- **Today**: Day 5 (Real inference structure implemented)
- **Version**: Preparing for v0.4.0 release

## What's Next (Immediate Priority Order)

### Day 5 (COMPLETING TODAY)
- [x] Implement real DiffSinger loader scaffolds
- [x] Implement fallback synthesis path
- [x] Test determinism with seed control
- [x] Verify speaker_id propagation
- [ ] **NEXT**: Validate that models load correctly when paths provided

### Days 6-7: Multi-Speaker & GPU Validation
**Objective**: Ensure real DiffSinger path works end-to-end

**Tasks**:
1. Acquire or create minimal DiffSinger checkpoint for testing
2. Provide DIFFSINGER_MODEL_PATH to container
3. Test real synthesis path engages (change synthesis_method from "fallback" to "diffsinger")
4. Validate multi-speaker support:
   - Create speaker embeddings for alice/bob/charlie
   - Verify different speakers produce different audio contours
   - Measure inference latency per speaker

5. Vocoder integration:
   - Load NSF-HiFiGAN checkpoint
   - Test mel-spec → audio pipeline
   - Validate 48kHz output quality

6. Performance benchmarking:
   - Measure single synthesis latency (target: <8s per 30s on GPU)
   - Test batch synthesis (4+ parallel requests)
   - Profile memory usage (target: <2GB GPU for batch)

### Days 8-9: Similarity Pod Setup
**Objective**: Begin Similarity Pod implementation (highest priority for MVP)

**Context**: Similarity Pod blocks export gating feature
- User uploads 30s reference audio
- System extracts features (pitch contour, rhythm, key)
- Compares against generated melody
- Returns verdict: pass/borderline/block

**Structure** (analogous to Voice Pod):
- `pods/similarity/main.py`: FastAPI service
- `pods/similarity/model.py`: Feature extractors + comparators
- `pods/similarity/g2p.py`: (Not needed, but feature alignment)

**Research Required**:
- Interval n-gram Jaccard similarity metric (librosa/numpy)
- DTW (Dynamic Time Warping) for rhythm comparison
- Key/BPM estimation from reference (librosa)
- Thresholds: pass <0.35, borderline 0.35-0.48, block ≥0.48

### Days 10-15: API Integration & Export
**Objective**: Connect Voice + Similarity → API workers → export pipeline

**Workers to Implement**:
- `VoiceSynthWorker`: Enqueue voice synthesis jobs, store results in S3
- `SimilarityWorker`: Load reference, compute features, run comparison
- `ExportWorker`: Compile stems, apply similarity verdict, create export bundle

**API Endpoints to Add**:
- `POST /remix/reference/upload`: Accept reference audio
- `POST /check/similarity`: Trigger similarity check (gating before export)
- `POST /export`: Final export with gating enforcement

## Decision Points

### Model Acquisition
**Question**: Where to get DiffSinger + vocoder checkpoints?

**Options**:
1. **OpenVoiceV2 Official Checkpoints** (if publicly available)
   - Pros: Production-quality, well-tested
   - Cons: May have licensing restrictions
   
2. **Community Checkpoints** (Hugging Face, GitHub)
   - Pros: Free, many variants
   - Cons: Variable quality, limited support
   
3. **Train Minimal Model** (for testing only)
   - Pros: Fully controlled, no licensing concerns
   - Cons: Time-intensive, requires good data

4. **Use Stub Longer** (continue fallback until real integration)
   - Pros: No blocker, can continue on other features
   - Cons: Missing validation on real model path

**Recommendation**: Continue with stubbed synthesis for now; real models can be plugged in once acquired. This unblocks Similarity Pod work (critical path).

### Similarity Algorithm Choice
**Question**: Interval n-gram or contour matching?

**Current Plan**: 
- Primary: Interval n-gram Jaccard (n=3,4,5)
- Secondary: DTW on normalized F0 curve
- Hard rule: 8+ bar near-identical melody always blocks

**Rationale**: Interval n-grams capture melodic "shape" without being sensitive to absolute pitch (user can shift key). DTW catches rhythmic similarity.

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| DiffSinger model unavailable | Blocks real synthesis validation | Use checkpoints from HF hub or cont. with stub |
| GPU memory exceeded | Performance failure | Profile early, implement streaming if needed |
| Similarity metrics too loose | Export gating ineffective | Use conservative thresholds, add hard rules |
| TTFP exceeds 45s | UX failure | Parallelize: voice + similarity in same worker |
| Seed non-determinism | Reproducibility broken | Test monthly with fixed corpus |

## Success Criteria (End of Sprint 3)

### Voice Pod
- ✅ Real synthesis path implemented (or stub acceptable for MVP)
- ✅ Determinism verified (same seed → same bytes)
- ✅ Multi-speaker support (3+ speakers tested)
- ✅ Performance: TTFP ≤45s P50 (including melody + voice)

### Similarity Pod
- ✅ Feature extraction working (pitch, rhythm, key)
- ✅ Similarity metric implemented (interval n-grams + DTW)
- ✅ Verdict thresholds validated (pass/borderline/block)
- ✅ API endpoint wired (POST /check/similarity)

### API Integration
- ✅ Voice worker in BullMQ (vocal queue)
- ✅ Similarity worker in BullMQ (check queue)
- ✅ Export worker with gating (export queue)
- ✅ End-to-end: lyrics → melody → voice → similarity check → export

### Release (v0.4.0)
- ✅ All above working in staging
- ✅ E2E test: Create song, regenerate section, remix with ref, export
- ✅ Performance test: TTFP ≤45s, peak memory <4GB
- ✅ Merge to main, tag v0.4.0, deploy to production

## Time Estimates (Remaining)

| Task | Days | Status |
|------|------|--------|
| Voice Pod validation (Days 6-7) | 2 | Not started |
| Similarity Pod implementation (Days 8-9) | 2 | Not started |
| API integration (Days 10-12) | 3 | Not started |
| Testing & hardening (Days 13-15) | 3 | Not started |
| **Total Remaining** | **10** | Ready to execute |

## Next Session Instructions

**Start with:**
1. Validate voice synthesis with real fallback (current state is good)
2. Plan Similarity Pod implementation (research interval n-grams)
3. Consider model acquisition strategy

**Don't block on:**
- Real DiffSinger checkpoint (fallback sufficient)
- GPU availability (CPU fallback works)
- Model training (use existing checkpoints if available)

**Do prioritize:**
- Similarity Pod research (critical path for v0.4.0)
- API worker structure (foundation for integration)
- Performance benchmarking (early warning for targets)

## References

- DiffSinger paper: `VOICE_POD_RESEARCH_DAY1.md`
- Current implementation: `VOICE_POD_DAY5_SUMMARY.md`
- Sprint plan: `docs/project/sprints/sprint_plan_s_0_s_1.md` (will update for S3)

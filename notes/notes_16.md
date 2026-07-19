# notes_16: Investigate notes_13's 0.64 Alignment Approach

**Date:** 2025-11-29
**Objective:** Understand how notes_13 achieved 0.64 avg alignment and replicate the approach
**Motivation:** notes_15 achieved only 0.36 with mixed-domain bridge; notes_13's 0.64 is proven to work

---

## Background

**notes_15 Results:**
- Mixed training (50 code + 50 English): 0.96 training, 0.36 benchmark
- Code-only training: 0.97 training, 0.16 benchmark
- **Gap from notes_13:** 0.36 vs 0.64 (1.78x difference)

**notes_13 Results (from FINAL_REPORT.md):**
- Average alignment: **0.64**
- Bridge used: `bridges/l0_code_bridge.pt`
- Same 7 benchmark queries as notes_15

---

## Research Questions

### 1. Bridge Architecture
- **Q:** What are the parameters of `bridges/l0_code_bridge.pt`?
  - Rank: 16? 32? 64?
  - Training samples: How many?
  - Source/target dimensions

### 2. Training Data
- **Q:** What data was `l0_code_bridge.pt` trained on?
  - Code-only? Prompt-only? Mixed?
  - How many samples?
  - Were the 7 benchmark queries included in training?

### 3. Training Method
- **Q:** How was the bridge trained?
  - Same Procrustes formula as notes_15?
  - Different preprocessing?
  - Different evaluation metric?

---

## Investigation Plan

### Phase 1: Bridge Archaeology (v's tasks)

**Task 1:** Inspect `bridges/l0_code_bridge.pt` metadata
```python
import torch
data = torch.load("bridges/l0_code_bridge.pt")
# Extract: rank, n_samples/fit_samples, source_dim, target_dim
```

**Task 2:** Search for training script
```bash
# Find where l0_code_bridge.pt was created
grep -r "l0_code_bridge" notes_*/
grep -r "save.*l0_code" notes_*/
```

**Task 3:** Find training data artifacts
```bash
# Look for training data files
ls -la notes_*/data/
ls -la notes_*/*training*.json
```

### Phase 2: Replication Test (v's tasks)

**Task 4:** Re-run notes_13 benchmark with current bridges
- Test `bridges/l0_code_bridge.pt` (notes_13 original)
- Test `bridges/granite_starcoder_mixed_r32.pt` (notes_15 mixed)
- Test `bridges/granite_starcoder_r32.pt` (notes_15 code-only)
- **Hypothesis:** If l0_code_bridge still gets 0.64, it's the bridge not the code

**Task 5:** Compare embedding extraction
- Verify notes_13 benchmark uses same extraction as notes_15
- Check: L0 averaging, no special tokens, truncation length

### Phase 3: Training Replication (if needed)

**Task 6:** If we find notes_13 training data, retrain bridge
- Use exact same data and parameters
- Validate we can reproduce 0.64 alignment

---

## Hypotheses

### Hypothesis A: Different Rank
- notes_13 used rank=64 (more capacity)
- notes_15 used rank=32
- **Test:** Check bridge metadata

### Hypothesis B: Prompt-Specialized Training
- notes_13 trained on 100+ English prompts ONLY
- notes_15 trained on 50/50 mix
- **Test:** Find training data

### Hypothesis C: More Training Samples
- notes_13 used 500+ samples
- notes_15 used 100 samples
- **Test:** Check n_samples in metadata

### Hypothesis D: Different Evaluation
- notes_13 calculated alignment differently
- notes_15 uses cosine in low-rank space
- **Test:** Compare benchmark code line-by-line

---

## Success Criteria

**Minimum Success:**
- Understand why notes_13 achieved 0.64
- Document the difference vs notes_15 approach

**Full Success:**
- Replicate notes_13 training process
- Achieve 0.64+ alignment with new bridge
- Document reproducible training recipe

---

## Timeline

**Phase 1 (Bridge Archaeology):** 2-4 hours
**Phase 2 (Replication Test):** 1-2 hours
**Phase 3 (Training Replication):** 3-4 hours (if needed)

**Total:** 6-10 hours (1-2 days)

---

## Files to Investigate

```
notes_13/
├── benchmark_tensorport.py     # Uses bridges/l0_code_bridge.pt (line 84)
├── results/
│   └── tensorport_results.json # 0.64 alignment results
└── [training scripts?]

bridges/
├── l0_code_bridge.pt          # The mystery bridge (0.64 alignment)
├── granite_starcoder_r32.pt   # notes_15 code-only (0.16 benchmark)
└── granite_starcoder_mixed_r32.pt # notes_15 mixed (0.36 benchmark)

notes_9/
└── embedding_bridge.py         # EmbeddingBridge class (check training history)
```

---

## Next Steps

1. ✅ Created notes_16.md plan
2. ⏳ Queue tasks for v in v_queue.md
3. ⏳ v investigates bridge metadata and training artifacts
4. ⏳ t analyzes findings and determines replication strategy
5. ⏳ v retrains bridge if needed
6. ⏳ t validates 0.64+ alignment achieved

---

**Status:** Awaiting v's archaeological investigation

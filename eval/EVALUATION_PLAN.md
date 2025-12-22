# 3DRAG Retrieval Evaluation Plan

## Goal
Create a labeled evaluation dataset to measure semantic search quality with realistic user queries.

## Dataset Specifications
- **Size**: 500 high-quality 3D models from Objaverse-LVIS (~820 MB)
- **Categories**: 50 LVIS categories (10 models each)
- **Queries**: 50 test queries (1 per category)
- **Labels**: 10 relevant models per query (all models in category)

---

## Steps

### Step 1: Select High-Quality Models ✅
- [x] Define 50 LVIS categories (vehicles, furniture, animals, etc.)
- [x] Source from Objaverse-LVIS (46k curated models)
- [x] Filter for size (≤5MB per model)
- [x] Download 345 models (~500MB total)
- [x] Export to `eval/models.json`

### Step 2: Process Models Through Pipeline
- [ ] Render preview images
- [ ] Generate captions with Florence-2
- [ ] Create embeddings with sentence-transformers
- [ ] Store in separate eval index

### Step 3: Create Query Dataset ✅
- [x] Write 50 realistic user queries (1 per category)
- [x] Include direct, descriptive, and use-case queries
- [x] Export to `eval/queries.json`

### Step 4: Label Ground Truth ✅
- [x] Auto-generated: all models in category are relevant
- [x] Export to `eval/ground_truth.json`
- [ ] (Optional) Refine manually for more precise labels

### Step 5: Implement Evaluation Script ✅
- [x] Load queries and ground truth
- [x] Query search API, get top-k results
- [x] Compute metrics:
  - MRR (Mean Reciprocal Rank)
  - Recall@1, @5, @10
  - NDCG@10
  - Precision@5
  - Per-category breakdown
- [x] Generate report with failure analysis

### Step 6: Baseline & Iterate
- [ ] Run evaluation on current system
- [ ] Document baseline metrics
- [ ] Identify failure cases
- [ ] Iterate on embeddings/prompts

---

## File Structure

```
eval/
├── EVALUATION_PLAN.md      # This file
├── models.json             # 100 curated models with metadata
├── queries.json            # 50 test queries
├── ground_truth.json       # Query → relevant model mappings
├── eval_index/             # Separate FAISS index for eval
│   ├── models.index
│   └── metadata.json
├── previews/               # Preview images for eval models
└── evaluate.py             # Evaluation script
```

---

## Categories (Draft)

1. **Vehicles** - cars, trucks, motorcycles, aircraft
2. **Furniture** - chairs, tables, sofas, beds
3. **Animals** - mammals, birds, fish, insects
4. **Characters** - humans, fantasy, robots
5. **Architecture** - buildings, houses, structures
6. **Nature** - trees, plants, rocks, terrain
7. **Food** - fruits, dishes, kitchen items
8. **Electronics** - computers, phones, appliances
9. **Weapons** - swords, guns, shields
10. **Sports** - equipment, balls, gear

---

## Metrics Targets

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| MRR | 0.3 | 0.5 | 0.7+ |
| Recall@5 | 0.4 | 0.6 | 0.8+ |
| NDCG@10 | 0.4 | 0.6 | 0.8+ |

---

## Timeline
- Step 1-2: Model selection and processing
- Step 3-4: Query creation and labeling (~1 hour manual work)
- Step 5-6: Evaluation and iteration

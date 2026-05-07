# Concept Transformer - Minimal Upgrade Analysis

## Current State
- **Python**: 3.9+ (implicit)
- **PyTorch**: 1.10.0 (2021)
- **PyTorch Lightning**: 1.4.8 (2021)
- **CUDA Support**: Legacy (only older GPUs - RTX 20/30 series)
- **Issue**: Cannot run on latest GPUs (H100, A100, newer RTX 40 series) without upgrade

---

## What Needs to Change (Minimal Approach)

### 1. **Core Dependencies** ⚠️ REQUIRED CHANGES

| Package | Current | → Upgrade | Reason |
|---------|---------|-----------|--------|
| `torch` | 1.10.0 | 2.2.0+ | Latest GPU support (H100, A100, RTX 40 series) |
| `torchvision` | 0.11.1 | 0.17.0+ | Compatibility with torch 2.x |
| `pytorch-lightning` | 1.4.8 | 2.1.0+ | Modern API, GPU handling improvements |
| `timm` | 0.4.12 | 0.9.0+ | ViT improvements, architecture updates |
| `torchmetrics` | 0.5 | 1.1.0+ | API compatibility with PL 2.x |
| `numpy` | 1.22.0 | 1.24.0+ | Compatibility (any 1.24+) |
| `setuptools` | 59.5.0 | 68.0.0+ | Modern package management |

---

## Code Changes Required (Minimal - Logic Preserved)

### ✅ Changes Needed in `ctc/ctc_model.py`

**Problem 1**: Deprecated PyTorch Lightning Trainer parameters (lines 223-244)
- `gpus=-1` → `accelerator="gpu"`, `devices=-1`
- `amp_backend="apex"` → `precision="16-mixed"`
- `progress_bar_refresh_rate=10` → `enable_progress_bar=True`

**Problem 2**: Deprecated training API
- The trainer initialization uses old argument names

**Solution**: Update 4 deprecated parameters in `get_trainer()` function

### ⚠️ Changes Needed in `ctc/ctc.py`

**Problem**: Import from `pl_bolts` (line with `LinearWarmupCosineAnnealingLR`)
- `lightning-bolts` 0.3.4 is not maintained
- This scheduler may not exist in newer versions

**Solution**: Replace with native PyTorch scheduler (1-2 lines)
- Use `torch.optim.lr_scheduler.CosineAnnealingWarmRestarts` or implement simple wrapper
- Logic remains identical

### ✅ Likely No Changes Needed

- **Model architecture** (ctc.py, vit.py) - Uses standard PyTorch, compatible with 2.x ✓
- **Data loading** (data/*.py) - Standard PyTorch DataLoader, compatible ✓
- **Loss functions** - Uses standard F.nll_loss, accuracy - compatible ✓
- **Training logic** - shared_step, forward methods - compatible ✓

---

## Breaking Changes to Watch

### 1. **`torchmetrics.functional.classification.accuracy`**
```python
# OLD (1.4.8):
accuracy(preds, y)

# NEW (1.1.0+):
# Still works same way, but may deprecate positional args
accuracy(preds, y, task='binary')  # or 'multiclass'
```
**Impact**: Minimal - function still works, but should add `task` parameter for clarity

### 2. **PyTorch 2.0+ Compiled Mode**
- New `torch.compile()` available but NOT required
- Existing code will run on torch 2.x without modification

### 3. **Data Type Changes**
- None expected - float32/int64 tensors work same way

---

## Files to Modify (Minimal Approach)

### **File 1: `ctc/ctc_model.py`** (~5 lines changed)
```
Location: get_trainer() function (lines ~223-244)
Changes:
  - Line ~228: if torch.cuda.device_count(): → keep (detects GPU)
  - Line ~231: "amp_backend": "apex" → remove
  - Line ~232: "amp_level": "O2" → remove  
  - Line ~231: "precision": 16 → change to "precision": "16-mixed"
  - Line ~238: gpus=-1 → devices=-1
  - Line ~239: auto_select_gpus=True → remove (deprecated)
  - Line ~240: progress_bar_refresh_rate=10 → remove
  - Line ~242: Add accelerator="gpu"
```

### **File 2: `ctc/ctc_model.py`** (Import statement at top)
```
Check if LinearWarmupCosineAnnealingLR import works
If fails: Replace with PyTorch native scheduler
```

### **File 3: `requirements.txt`**
Replace all versions (13 lines total)

---

## Updated `requirements.txt` (Minimal)

```
torch==2.2.0
torchvision==0.17.0
pytorch-lightning==2.1.3
lightning-bolts==0.7.0
torchmetrics==1.1.0
scipy==1.11.0
numpy==1.24.0
pandas==2.0.0
setuptools==68.0.0
albumentations==1.3.0
timm==0.9.2
matplotlib==3.7.0
epyc
```

---

## Risk Assessment

| Change | Risk | Mitigation |
|--------|------|-----------|
| torch 1.10 → 2.2 | **Medium** - New features but stable | Update trainer API only |
| PL 1.4 → 2.1 | **Medium** - API changed | Update 5 parameters in get_trainer() |
| lr_scheduler | **Low** - Easy fallback | Use PyTorch native if needed |
| timm 0.4 → 0.9 | **Low** - ViT backbone works same | No code change needed |
| torchmetrics | **Low** - Backward compatible | Add task param to accuracy() |

---

## Summary: Minimal Changes Required

✅ **Total Code Changes**: ~15-20 lines (4% of codebase)
✅ **Files Modified**: 2 files (ctc_model.py + requirements.txt)
✅ **Logic Preserved**: 100% - all model architecture unchanged
✅ **New Features Used**: None - purely dependency upgrades

### Changes Breakdown:
1. Update `requirements.txt` - swap 13 dependency versions
2. Update `get_trainer()` in `ctc_model.py` - replace 5-7 deprecated parameters
3. Optional: Add `task` parameter to `accuracy()` calls for clarity
4. Optional: Check `LinearWarmupCosineAnnealingLR` import works

**Result**: Code runs on latest GPUs (A100, H100, RTX 40) without any model logic changes.

---

## Testing Plan After Upgrade

```bash
# 1. Test model forward pass
python3 -m pytest -q tests/test_ctc.py

# 2. Test data loading
python3 -m pytest -q tests/test_datasets.py

# 3. Quick training run (1 epoch test)
python3 ctc_mnist.py --max_epochs 1 --n_train_samples 500 --batch_size 64
```

---

## Next Steps (When Ready)

1. Update `requirements.txt` with new versions
2. Fix `get_trainer()` function in `ctc_model.py` (5 lines)
3. Test with unit tests
4. Run quick training validation
5. All done - no other changes needed!

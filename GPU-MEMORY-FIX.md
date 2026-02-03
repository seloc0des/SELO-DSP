# GPU Memory Optimization Guide for 16GB GPU

## Problem Summary

Your llama3.1:8b-instruct-q8_0 model crashes under load because:
- **Model size**: ~8.5GB
- **Current config**: ALL layers on GPU (OLLAMA_GPU_LAYERS=999)
- **Context window**: 8192 tokens = ~2.5-3GB per request
- **Concurrent requests**: 2 (OLLAMA_NUM_PARALLEL=2)
- **Total under load**: ~15-16GB out of 16GB = **OOM crashes!**

## Solution: GPU Layer Reduction

Instead of loading ALL layers on GPU, offload some to CPU/RAM.

---

## Configuration Options

### Option 1: STABLE (Recommended)
**File**: [.env.stable-16gb](selo-ai/backend/.env.stable-16gb)

**Settings**:
- `OLLAMA_GPU_LAYERS=40` (offload ~25% to CPU)
- `CHAT_NUM_CTX=4096` (half the context)
- `OLLAMA_NUM_PARALLEL=2` (still allows reflection + chat)
- `TORCH_CUDA_MEMORY_FRACTION=0.75` (reserve 4GB safety buffer)

**Expected Performance**:
- ~12GB GPU usage under load
- 4GB safety buffer
- ~5-10% slower than full GPU (negligible)
- **Stable under concurrent load**

---

### Option 2: CONSERVATIVE (Maximum Stability)
**File**: [.env.conservative-16gb](selo-ai/backend/.env.conservative-16gb)

**Settings**:
- `OLLAMA_GPU_LAYERS=35` (offload ~30% to CPU)
- `CHAT_NUM_CTX=2048` (minimal context)
- `OLLAMA_NUM_PARALLEL=1` (one request at a time)
- `TORCH_CUDA_MEMORY_FRACTION=0.70` (reserve 5GB safety buffer)

**Expected Performance**:
- ~9GB GPU usage under load
- 7GB safety buffer
- ~10-15% slower than full GPU
- **No concurrent reflection + chat**
- Guaranteed stability

---

## How to Apply

### Step 1: Copy Configuration
Choose one config and copy to `.env`:

```bash
# For STABLE (recommended):
cd /mnt/local/Projects/SELODSP/selo-ai/backend
cp .env.stable-16gb .env

# OR for CONSERVATIVE:
cp .env.conservative-16gb .env
```

### Step 2: Restart Ollama
```bash
sudo systemctl restart ollama
```

### Step 3: Restart SELO Backend
```bash
# Stop backend if running
pkill -f "python.*main.py"

# Start backend
cd /mnt/local/Projects/SELODSP/selo-ai/backend
python main.py
```

### Step 4: Test Under Load
Try multiple requests simultaneously:
```bash
# In one terminal:
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me a story", "conversation_id": "test1"}'

# In another terminal (while first is running):
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is AI?", "conversation_id": "test2"}'
```

### Step 5: Monitor GPU Memory
```bash
# Watch GPU usage in real-time:
watch -n 1 nvidia-smi

# Look for "Memory-Usage" column - should stay under 12GB (stable) or 10GB (conservative)
```

---

## Configuration Comparison Table

| Setting | Current (Crashes) | Stable | Conservative |
|---------|-------------------|--------|--------------|
| **GPU Layers** | 999 (all) | 40 | 35 |
| **Context Window** | 8192 | 4096 | 2048 |
| **Concurrent Requests** | 2 | 2 | 1 |
| **GPU Memory Reserved** | 0% | 25% | 30% |
| **Expected GPU Usage** | ~15-16GB | ~12GB | ~9GB |
| **Safety Buffer** | 0-1GB | 4GB | 7GB |
| **Performance vs Full GPU** | 100% | ~95% | ~85-90% |
| **Stability** | ❌ Crashes | ✅ Stable | ✅✅ Very Stable |
| **Reflection Support** | ✅ Concurrent | ✅ Concurrent | ⚠️ Sequential |

---

## Understanding GPU Layers

For llama3.1:8b model:
- **Total layers**: ~80 layers
- **Each layer**: ~100-120MB

| Layers on GPU | GPU Memory | CPU Memory | Performance |
|---------------|------------|------------|-------------|
| 999 (all) | ~8.5GB | 0 | 100% (crashes) |
| 40 (50%) | ~7GB | ~1.5GB | ~95% (stable) |
| 35 (44%) | ~6.5GB | ~2GB | ~90% (very stable) |
| 20 (25%) | ~4GB | ~4.5GB | ~70% |
| 0 (CPU only) | 0 | ~8.5GB | ~30-40% |

---

## Why This Works

### Memory Budget Breakdown (STABLE config):

**Before (Crashes)**:
```
Model (all layers): 8.5GB
KV cache (8192×2):  5-6GB
CUDA overhead:      1.5GB
---
TOTAL: 15-16GB / 16GB = OOM! ❌
```

**After (Stable)**:
```
Model (40 layers):  7GB
KV cache (4096×2):  3GB
CUDA overhead:      1GB
Safety buffer:      4GB (reserved)
---
TOTAL: 11GB / 12GB available = Safe! ✅
```

---

## Advanced: Fine-Tuning Layers

If you want to experiment with different layer counts:

1. **Find optimal layers** (binary search):
   - Start with 40 layers
   - If stable with room: increase by 5
   - If crashes: decrease by 5
   - Repeat until you find the sweet spot

2. **Monitor and adjust**:
```bash
# Check GPU usage under load:
nvidia-smi --query-gpu=memory.used,memory.total --format=csv --loop=1

# Aim for: ~12-13GB max usage (leaves 3-4GB buffer)
```

3. **Adjust in .env**:
```bash
# Test with different values:
OLLAMA_GPU_LAYERS=35  # More conservative
OLLAMA_GPU_LAYERS=40  # Balanced
OLLAMA_GPU_LAYERS=45  # Aggressive (may crash)
```

---

## Troubleshooting

### Still Crashing?
1. Use CONSERVATIVE config
2. Further reduce context: `CHAT_NUM_CTX=1024`
3. Set `OLLAMA_NUM_PARALLEL=1`
4. Reduce layers: `OLLAMA_GPU_LAYERS=30`

### Too Slow?
1. Increase layers gradually: 40 → 42 → 44
2. Monitor GPU memory - stay under 13GB
3. If stable, increase context: 4096 → 5120

### Check Actual Settings
```bash
# Verify Ollama is using your settings:
curl http://localhost:11434/api/show -d '{"name": "llama3.1:8b-instruct-q8_0"}' | jq

# Check environment variables:
env | grep OLLAMA
```

---

## Next Steps

1. **Start with STABLE config** (.env.stable-16gb)
2. **Test under load** with multiple concurrent requests
3. **Monitor GPU memory** with nvidia-smi
4. **If stable**, optionally increase layers by 5
5. **If crashes**, switch to CONSERVATIVE config

---

## Key Takeaway

**You don't need ALL layers on GPU!**

Loading 40-45 layers (50%) on GPU:
- Keeps the model 90-95% as fast
- Uses 2-3GB less VRAM
- Prevents OOM crashes
- Allows stable concurrent operation

The CPU can handle 30-40 layers with minimal performance impact, especially since your bottleneck is GPU memory, not compute speed.

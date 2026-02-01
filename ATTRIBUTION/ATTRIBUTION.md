# Code Attribution

## Our Core Contributions (To Be Implemented)

The following files contain our original work and core innovations:

- `nano_dh/hybrid_model.py` - **TODO**: 3:1 GatedDeltaNet:GatedAttention hybrid architecture
- `nano_dh/deltanet.py` - **TODO**: GatedDeltaNet layer implementation
- `nano_dh/gated_attn.py` - **TODO**: GatedAttention layer implementation
- Benchmarks and scaling law experiments for hybrid architectures

## Infrastructure Code (From nanochat)

The following files are copied from [nanochat](https://github.com/karpathy/nanochat) (MIT License, Copyright 2025 Andrej Karpathy):

### Core Infrastructure (`nano_dh/`)
- `tokenizer.py` - BPE tokenizer implementation
- `dataloader.py` - BOS-aligned data loading
- `dataset.py` - Dataset utilities
- `checkpoint_manager.py` - Model checkpoint save/load
- `common.py` - Distributed training utilities
- `optim.py` - Muon + AdamW optimizers
- `engine.py` - Inference engine with KV cache (will need adaptation for DeltaNet)
- `flash_attention.py` - Flash Attention wrapper (used for Attention layers)
- `core_eval.py` - CORE metric evaluation
- `loss_eval.py` - Bits-per-byte evaluation
- `report.py` - Training report utilities
- `execution.py` - Python code execution for LLM
- `ui.html` - Web chat interface
- `gpt_baseline.py` - Original GPT implementation (kept as reference/baseline)

### Training Scripts (`scripts/`)
- `base_train.py` - Pretraining script (will be adapted)
- `base_eval.py` - Base model evaluation
- `chat_sft.py` - Supervised fine-tuning
- `chat_rl.py` - Reinforcement learning
- `chat_cli.py` - CLI chat interface
- `chat_web.py` - Web chat server
- `chat_eval.py` - Chat model evaluation
- `tok_train.py` - Tokenizer training
- `tok_eval.py` - Tokenizer evaluation

### Evaluation Tasks (`tasks/`)
- `arc.py`, `gsm8k.py`, `humaneval.py`, `mmlu.py`, `smoltalk.py`, `spellingbee.py`, `customjson.py`, `common.py`

### Test Files (`tests/`)
- All test files from nanochat

### Run Scripts (`runs/`)
- Shell scripts for training experiments

## Third-Party Libraries

- **flash-linear-attention** (MIT License, Copyright 2023-2025 Songlin Yang)
  - Used for: Efficient DeltaNet kernels
  - See: [ATTRIBUTION/FLA.LICENSE](ATTRIBUTION/FLA.LICENSE)

## Development Roadmap

### Phase 1: Quick Prototype (Week 1-2)
- [x] Copy nanochat infrastructure
- [ ] Implement `nano_dh/deltanet.py` using FLA kernels
- [ ] Implement `nano_dh/gated_attn.py`
- [ ] Implement `nano_dh/hybrid_model.py` with 3:1 mixing strategy
- [ ] Adapt `scripts/base_train.py` to support hybrid model
- [ ] Verify training runs and basic functionality

### Phase 2: Experiments & Optimization (Month 1-2)
- [ ] Run scaling laws experiments for hybrid architecture
- [ ] Benchmark against nanochat baseline
- [ ] Optimize DeltaNet layer efficiency
- [ ] Long-context evaluations (32K+)
- [ ] Memory profiling and optimization

### Phase 3: Independent Infrastructure (Future)
- [ ] Simplify dataloader (if needed)
- [ ] Custom KV cache implementation for hybrid model
- [ ] Streamline training scripts
- [ ] Remove unused nanochat code
- [ ] Comprehensive documentation

## Why We Build on nanochat

nanochat provides a solid, well-tested, and hackable foundation for LLM training. 
Instead of reinventing infrastructure, we focus our efforts on our core innovation: 
efficient hybrid architectures combining linear and quadratic attention mechanisms.

This approach allows us to:
1. Launch quickly and validate our core ideas
2. Benefit from nanochat's optimizations and community testing
3. Focus development time on novel contributions
4. Provide an easy migration path for nanochat users

See [ATTRIBUTION/NANOCHAT.LICENSE](ATTRIBUTION/NANOCHAT.LICENSE) for full license terms.

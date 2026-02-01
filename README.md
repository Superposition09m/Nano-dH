# Nano-∆H

![Nano-∆H](misc/figs/nano-dH.png)

Nano-∆H is an efficient hybrid LM(3:1 GatedDeltaNet:GatedAttention) that everyone can train.

## Architecture

```
Layer 0:  DeltaNet   ←─┐
Layer 1:  DeltaNet     ├─ 75% linear complexity
Layer 2:  DeltaNet   ←─┘
Layer 3:  Attention  ←── 25% for long-range
[repeat...]
```



## File structure

```
Nano-dH/
├── nano-dh/
├── data/
├── scripts/
├── tasks/
├── tokenizer_bpe/
└── tests/
```


## Acknowledgments

- This project is inspired by [nanochat](https://github.com/karpathy/nanochat) and [Qwen3-Next](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list).

- And also thanks to [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) for the efficient kernels.
# awq-qwen-toolcall

Activation-Aware Weight Quantization (AWQ) applied to Qwen3.5-2B for tool-calling tasks.  
Built for local experimentation on Apple Silicon (MPS).

## Overview

This repo implements AWQ from scratch in pure PyTorch to study how INT4 weight quantization affects a model's ability to generate structured tool calls. All experiments run fully offline on Mac hardware.

## Structure

```
├── awq/               # AWQ quantizer (calibrate → scale → quantize)
├── eval/              # Evaluation harness + benchmarks
├── results/           # Outputs: quantized weights, benchmarks, diffs
├── models/            # Local model weights (gitignored)
├── data/              # Calibration + evaluation prompts (ToolACE)
└── README.md
```

## Use Case: Tool-Calling

Evaluated against [ToolACE](https://huggingface.co/datasets/lockon/ToolACE) — 11,300 tool-calling conversations with structured function definitions, parameters, and expected outputs.

**Quality metrics:**
- Function call validity (JSON syntax)
- Function name match rate
- Required parameter presence
- Semantic similarity (embedding-based)
- Per-token throughput (MPS)
- Peak memory usage

## License

MIT

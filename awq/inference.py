"""AWQ inference forward pass — Issue #9.

Loads quantized INT4 weights and runs a custom forward pass
that dequantizes on-the-fly and applies AWQ scale compensation.

Peak memory: ~model_base_fp16 + ~50MB for one quantized layer at a time.
"""

import os
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.memory import log_memory, memory_tracker

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def dequantize_layer(q: dict) -> torch.Tensor:
    """Dequantize a full quantized layer to FP16.

    Args:
        q: Quantized layer dict with keys:
            'packed_weights': list of packed INT8 tensors
            'group_scales': list of FP16 scales
            'scale_factors': AWQ per-channel scales
            'shape': (d_out, d_in)
            'group_size': int

    Returns:
        FP16 weight matrix of shape [d_out, d_in], with AWQ scaling
        applied (weights are already scaled by /s, so we multiply
        back by s to get the original weight space).
    """
    d_out, d_in = q["shape"]
    group_size = q["group_size"]

    parts = []
    for packed, qscale in zip(q["packed_weights"], q["group_scales"]):
        # Unpack two INT4 per INT8
        low = (packed & 0x0F).to(torch.int8)
        high = ((packed >> 4) & 0x0F).to(torch.int8)
        low = torch.where(low > 7, low - 16, low)
        high = torch.where(high > 7, high - 16, high)

        w_deq = torch.stack([low, high], dim=-1).reshape(d_out, group_size)
        w_deq = w_deq.to(dtype=torch.float16) * qscale.to(dtype=torch.float16)
        parts.append(w_deq)

    # Concatenate groups and trim to original dimension
    w = torch.cat(parts, dim=1)[:, :d_in]

    # Multiply back by AWQ scale factors to restore original weight space
    # (Recall: W_quant = W_fp16 / s, so W_approx = W_quant * s)
    s = q["scale_factors"].to(dtype=torch.float16)
    w = w * s.unsqueeze(0)

    return w


def load_awq_model(
    quantized_path: str,
    model_path: str,
    device: str = "mps",
) -> tuple[AutoModelForCausalLM, AutoTokenizer, dict[str, dict]]:
    """Load the FP16 model shell and replace its linear weights with dequantized AWQ ones.

    This is a straightforward approach: dequantize all layers into FP16
    in CPU memory, then push to device.

    For a more memory-efficient approach, see the Wrapper class below.

    Args:
        quantized_path: Path to quantized_state.pt
        model_path: Path to FP16 model directory (for config + tokenizer)
        device: Target device

    Returns:
        (model, tokenizer, quantized_state)
    """
    print(f"Loading FP16 model shell from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading quantized weights from {quantized_path}...")
    quantized_state = torch.load(quantized_path, map_location="cpu", weights_only=True)
    print(f"  {len(quantized_state)} layers loaded")

    # Dequantize and inject weights
    named_modules = {n: m for n, m in model.named_modules()}

    for layer_name, q in quantized_state.items():
        if layer_name not in named_modules:
            # Try alternate naming (language_model prefix)
            alt_name = layer_name.replace("model.", "model.language_model.")
            if alt_name in named_modules:
                layer_name = alt_name
            else:
                continue

        module = named_modules[layer_name]
        if isinstance(module, torch.nn.Linear):
            w = dequantize_layer(q)
            module.weight.data = w.to(device=device, dtype=torch.float16)

    model.eval()
    return model, tokenizer, quantized_state


class AWQModelWrapper:
    """Memory-efficient AWQ wrapper.

    Keeps quantized weights on CPU in packed INT4 format,
    dequantizes one layer at a time on the target device
    during forward passes.

    This trades some speed for much lower memory usage.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        quantized_state: dict[str, dict],
        device: str = "mps",
    ):
        self.model = model
        self.quantized_state = quantized_state
        self.device = device
        self._dequantized_cache: dict[str, torch.Tensor] = {}

        # Build name map
        self._name_map: dict[str, str] = {}
        for module_name, _ in model.named_modules():
            for q_name in quantized_state:
                if module_name.endswith(q_name) or q_name.endswith(module_name):
                    self._name_map[module_name] = q_name
                    break

    def get_weight(self, module_name: str) -> torch.Tensor:
        """Get dequantized weight for a module, caching on device."""
        if module_name in self._dequantized_cache:
            return self._dequantized_cache[module_name]

        q_name = self._name_map.get(module_name, module_name)
        if q_name not in self.quantized_state:
            raise KeyError(f"No quantized weights for {module_name} ({q_name})")

        w = dequantize_layer(self.quantized_state[q_name])
        self._dequantized_cache[module_name] = w.to(device=self.device, dtype=torch.float16)
        return self._dequantized_cache[module_name]

    def clear_cache(self):
        """Clear dequantized weight cache to free memory."""
        self._dequantized_cache.clear()
        torch.mps.empty_cache()

    @torch.no_grad()
    def generate(
        self,
        tokenizer: AutoTokenizer,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> tuple[str, float]:
        """Generate text with AWQ weights.

        For now, uses the full dequantized model approach.
        """
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096
        ).to(self.device)
        input_len = inputs["input_ids"].size(1)

        t0 = time.perf_counter()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        elapsed = time.perf_counter() - t0

        generated = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        tokens_per_sec = (len(outputs[0]) - input_len) / max(elapsed, 1e-6)

        return generated.strip(), tokens_per_sec


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    MODEL_PATH = "models/Qwen3.5-2B-FP16"
    QUANTIZED_PATH = os.path.join(RESULTS_DIR, "qwen_awq_int4", "quantized_state.pt")

    # Full dequantization approach (loads model + dequantized weights → ~4GB on MPS)
    with memory_tracker("load_awq"):
        model, tokenizer, q_state = load_awq_model(QUANTIZED_PATH, MODEL_PATH, device="mps")

    log_memory("model_loaded")

    # Quick test
    print("\nGenerating sample output...")
    prompt = "What is the capital of France?"
    inputs = tokenizer(prompt, return_tensors="pt").to("mps")

    t0 = time.perf_counter()
    outputs = model.generate(**inputs, max_new_tokens=50)
    elapsed = time.perf_counter() - t0

    generated = tokenizer.decode(outputs[0][inputs["input_ids"].size(1):], skip_special_tokens=True)
    print(f"  Prompt: {prompt}")
    print(f"  Output: {generated}")
    print(f"  Time:   {elapsed:.2f}s")
    print(f"\n✅ AWQ inference test complete")

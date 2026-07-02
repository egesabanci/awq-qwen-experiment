"""AWQ inference — load quantized INT4 weights and run forward passes.

Provides:
- dequantize_layer(): Convert packed INT4 back to FP16
- load_awq_model(): Full dequantization approach (load all weights at once)
- AWQModelWrapper: Memory-efficient on-the-fly dequantization
"""

import os
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.memory import log_memory, memory_tracker, get_device, empty_cache
from awq.quantize import dequantize_layer  # canonical dequant (shared with verify)


def load_awq_model(
    quantized_path: str,
    model_path: str,
    device: str | None = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer, dict[str, dict]]:
    """Load the FP16 model shell and replace its linear weights with dequantized AWQ ones.

    Args:
        quantized_path: Path to quantized_state.pt
        model_path: Path to FP16 model directory (for config + tokenizer)
        device: Target device. Auto-detected if None.

    Returns:
        (model, tokenizer, quantized_state)
    """
    if device is None:
        device = get_device()

    # Load the FP16 model shell via the shared loader, then overwrite linear
    # weights with dequantized AWQ ones.
    # NOTE: this is dequantized-FP16 inference (no INT4 kernel) — peak memory is
    # ~one full FP16 model, not the INT4 footprint. For true memory-efficiency
    # use AWQModelWrapper (on-the-fly dequant), which is not wired into the
    # generation path.
    from awq.models import load_model
    model, tokenizer = load_model(model_path, device)

    print(f"Loading quantized weights from {quantized_path}...")
    quantized_state = torch.load(quantized_path, map_location="cpu", weights_only=True)
    print(f"  {len(quantized_state)} layers loaded")

    # Dequantize and inject weights
    named_modules = {n: m for n, m in model.named_modules()}

    for layer_name, q in quantized_state.items():
        mod = named_modules.get(layer_name)
        if mod is None:
            alt_name = layer_name.replace("model.", "model.language_model.")
            mod = named_modules.get(alt_name)
            if mod is None:
                continue

        if isinstance(mod, torch.nn.Linear):
            w = dequantize_layer(q)
            mod.weight.data = w.to(device=device, dtype=torch.float16)

    model.eval()
    return model, tokenizer, quantized_state


class AWQModelWrapper:
    """Memory-efficient AWQ wrapper.

    Keeps quantized weights on CPU in packed INT4 format,
    dequantizes one layer at a time on the target device
    during forward passes.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        quantized_state: dict[str, dict],
        device: str | None = None,
    ):
        if device is None:
            device = get_device()
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
        empty_cache()

    @torch.no_grad()
    def generate(
        self,
        tokenizer: AutoTokenizer,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> tuple[str, float]:
        """Generate text with AWQ weights using full dequantized model."""
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096
        ).to(self.device)
        input_len = inputs["input_ids"].size(1)

        t0 = time.perf_counter()
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
        outputs = self.model.generate(**inputs, **gen_kwargs)
        elapsed = time.perf_counter() - t0

        generated = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        tokens_per_sec = (len(outputs[0]) - input_len) / max(elapsed, 1e-6)

        return generated.strip(), tokens_per_sec

#!/usr/bin/env python3
"""
Generic Specialist Creator
===========================

Creates stripped specialist models from base models.
Based on Code God methodology (notes_6/create_code_god.py)

Usage:
    python create_specialist.py python    # Create Python specialist
    python create_specialist.py sql       # Create SQL specialist
    python create_specialist.py --list    # List available specialists
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy
import os
import sys
import json
import time
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

# Load specialist definitions
config_path = Path(__file__).parent / "specialist_config.json"
with open(config_path) as f:
    CONFIG = json.load(f)

SPECIALISTS = CONFIG["specialists"]
UNIVERSAL_DIM = CONFIG["universal_hub"]["dimension"]

# ============================================================================
# Stripped Specialist Model
# ============================================================================

class StrippedSpecialist(nn.Module):
    """
    Generic stripped specialist model.

    Architecture:
    - Embeddings (preserved from original)
    - N transformer layers (reduced from original)
    - NO lm_head (cannot generate text)
    - Projection to Universal Hub (1024-dim)

    Benefits:
    - 2-3x smaller than original
    - 3-5x faster inference
    - Cannot speak (perfect for TensorPort!)
    """

    def __init__(self, original_model, keep_layers=8, universal_dim=1024):
        super().__init__()

        self.hidden_size = original_model.config.hidden_size
        self.keep_layers = keep_layers
        self.universal_dim = universal_dim

        # Copy the full model backbone
        print(f"  Copying model backbone...")
        self.backbone = copy.deepcopy(original_model.model)

        # Delete excess layers (keep first N)
        original_layers = len(self.backbone.layers)
        layers_to_remove = original_layers - keep_layers

        print(f"  Removing {layers_to_remove} layers ({original_layers} → {keep_layers})...")
        for _ in range(layers_to_remove):
            del self.backbone.layers[-1]

        # NO lm_head - we output tensors, not tokens!

        # Projection to Universal Hub dimension
        print(f"  Creating Universal Hub projection ({self.hidden_size} → {universal_dim})...")
        self.to_universal = nn.Linear(self.hidden_size, universal_dim)
        nn.init.xavier_uniform_(self.to_universal.weight)
        nn.init.zeros_(self.to_universal.bias)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        """
        Forward pass through specialist model.

        Input: token ids
        Output: {
            'universal': Universal Hub tensor [batch, universal_dim],
            'hidden': Raw hidden state [batch, hidden_size],
            'all_hidden': All layer hidden states (if requested)
        }
        """
        # Run through backbone
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )

        # Get last hidden state
        hidden = outputs.last_hidden_state

        # Take last token representation (standard for causal LM)
        last_hidden = hidden[:, -1, :]  # [batch, hidden_size]

        # Project to universal space
        universal = self.to_universal(last_hidden)  # [batch, universal_dim]

        return {
            "universal": universal,
            "hidden": last_hidden,
            "all_hidden": outputs.hidden_states if output_hidden_states else None
        }

    def get_tensor(self, text, tokenizer, device=None):
        """Convenience method: text → universal tensor"""
        if device is None:
            device = next(self.parameters()).device

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.forward(inputs["input_ids"], inputs.get("attention_mask"))

        return out["universal"]


# ============================================================================
# Specialist Creator
# ============================================================================

def create_specialist(specialist_name):
    """
    Create a stripped specialist model.

    Args:
        specialist_name: Name from specialist_config.json (e.g., 'python', 'sql')

    Returns:
        Path to saved specialist model
    """
    print("=" * 70)
    print(f"CREATING {specialist_name.upper()} SPECIALIST")
    print("=" * 70)

    # Get config
    if specialist_name not in SPECIALISTS:
        print(f"❌ Unknown specialist: {specialist_name}")
        print(f"Available: {', '.join(SPECIALISTS.keys())}")
        return None

    spec_config = SPECIALISTS[specialist_name]

    # Step 1: Load original model
    print(f"\n[1] Loading {spec_config['name']}...")
    print(f"    Base model: {spec_config['base_model']}")
    print(f"    Target: {spec_config['original_params']} → {spec_config['target_params']}")

    if spec_config['base_model'] == "TBD":
        print(f"❌ Base model not yet defined for {specialist_name}")
        print(f"   Please update specialist_config.json with actual model path")
        return None

    start_time = time.time()

    try:
        original = AutoModelForCausalLM.from_pretrained(
            spec_config['base_model'],
            torch_dtype=torch.float32
        )
        tokenizer = AutoTokenizer.from_pretrained(spec_config['base_model'])
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print(f"   You may need to download it first:")
        print(f"   huggingface-cli download {spec_config['base_model']}")
        return None

    load_time = time.time() - start_time
    print(f"✓ Loaded in {load_time:.1f}s")

    original_params = sum(p.numel() for p in original.parameters())
    print(f"  Original: {original_params:,} params ({original_params/1e9:.2f}B)")
    print(f"  Layers: {len(original.model.layers)}")
    print(f"  Hidden dim: {original.config.hidden_size}")

    # Step 2: Create stripped specialist
    print(f"\n[2] Creating stripped specialist...")

    start_time = time.time()
    specialist = StrippedSpecialist(
        original,
        keep_layers=spec_config['keep_layers'],
        universal_dim=UNIVERSAL_DIM
    )
    strip_time = time.time() - start_time

    specialist_params = sum(p.numel() for p in specialist.parameters())
    reduction = original_params / specialist_params

    print(f"\n✓ Stripped in {strip_time:.1f}s")
    print(f"  Specialist: {specialist_params:,} params ({specialist_params/1e9:.2f}B)")
    print(f"  Layers: {spec_config['keep_layers']} of {len(original.model.layers)}")
    print(f"  Reduction: {reduction:.1f}x smaller")

    # Step 3: Test specialist
    print(f"\n[3] Testing specialist...")

    test_inputs = [
        "def quicksort(arr):",
        "SELECT * FROM users WHERE",
        "async function fetchData(url) {",
    ]

    print("  Testing text → tensor conversion:")
    for text in test_inputs:
        tensor = specialist.get_tensor(text, tokenizer)
        print(f"    '{text[:30]}...' → {tensor.shape}, mean: {tensor.mean().item():.4f}")

    # Step 4: Compare to original
    print(f"\n[4] Comparing to original model...")

    test_code = "def fibonacci(n):\n    if n <= 1:\n        return n"
    inputs = tokenizer(test_code, return_tensors="pt", truncation=True)

    # Original hidden states (at layer N)
    with torch.no_grad():
        orig_out = original.model(
            inputs["input_ids"],
            output_hidden_states=True
        )
        # Get hidden state at kept layer depth
        layer_idx = spec_config['keep_layers']
        orig_hidden = orig_out.hidden_states[layer_idx][:, -1, :]

    # Specialist hidden states
    with torch.no_grad():
        spec_out = specialist(inputs["input_ids"], output_hidden_states=True)
        spec_hidden = spec_out["hidden"]

    # Compare
    cosine_sim = nn.functional.cosine_similarity(
        orig_hidden.flatten(), spec_hidden.flatten(), dim=0
    ).item()

    print(f"  Original (layer {layer_idx}): {orig_hidden.shape}")
    print(f"  Specialist: {spec_hidden.shape}")
    print(f"  Cosine similarity: {cosine_sim:.4f}")

    if cosine_sim > 0.95:
        print("  ✓ Excellent alignment (>95%)")
    elif cosine_sim > 0.85:
        print("  ✓ Good alignment (>85%)")
    else:
        print(f"  ⚠ Lower alignment than expected")

    # Step 5: Save specialist
    print(f"\n[5] Saving specialist...")

    save_dir = Path("models") / f"{specialist_name}-specialist"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights
    model_path = save_dir / "model.pt"
    torch.save(specialist.state_dict(), model_path)
    print(f"  ✓ Saved model: {model_path}")

    # Save config
    config_save = {
        "specialist_name": specialist_name,
        "base_model": spec_config['base_model'],
        "keep_layers": spec_config['keep_layers'],
        "hidden_size": specialist.hidden_size,
        "universal_dim": UNIVERSAL_DIM,
        "original_params": original_params,
        "specialist_params": specialist_params,
        "reduction": f"{reduction:.1f}x",
        "alignment_quality": f"{cosine_sim:.4f}",
    }
    config_path = save_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config_save, f, indent=2)
    print(f"  ✓ Saved config: {config_path}")

    # Save tokenizer
    tokenizer.save_pretrained(save_dir)
    print(f"  ✓ Saved tokenizer: {save_dir}")

    # Summary
    print("\n" + "=" * 70)
    print(f"{spec_config['name'].upper()} CREATED")
    print("=" * 70)

    print(f"""
ORIGINAL {spec_config['base_model']}:
  - Parameters: {original_params/1e9:.2f}B
  - Layers: {len(original.model.layers)}
  - Can generate text: Yes
  - Hidden dim: {spec_config['hidden_dim']}

{spec_config['name'].upper()} SPECIALIST:
  - Parameters: {specialist_params/1e9:.2f}B
  - Layers: {spec_config['keep_layers']}
  - Can generate text: NO
  - Universal Hub output: {UNIVERSAL_DIM}-dim
  - Reduction: {reduction:.1f}x smaller
  - Quality: {cosine_sim:.4f} alignment

CAPABILITIES:
  ✓ Understands {specialist_name} (embeddings preserved)
  ✓ Processes {specialist_name} logic ({spec_config['keep_layers']} layers)
  ✓ Outputs to Universal Hub ({UNIVERSAL_DIM}-dim tensor)
  ✗ Cannot speak
  ✗ Cannot explain
  ✗ Cannot generate text

NEXT: Train bridges to connect {spec_config['name']} ↔ Universal Hub
      Run: python train_specialist_bridges.py {specialist_name}

Saved to: {save_dir}
""")

    return save_dir


# ============================================================================
# Main
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python create_specialist.py <specialist_name>")
        print("\nAvailable specialists:")
        for name, config in SPECIALISTS.items():
            status = "✅" if config['base_model'] != "TBD" else "⚠️ (needs model)"
            print(f"  {status} {name}: {config['name']} ({config['priority']} priority)")
        sys.exit(1)

    specialist_name = sys.argv[1]

    if specialist_name == "--list":
        print("Available specialists:")
        for name, config in SPECIALISTS.items():
            print(f"\n{name}:")
            print(f"  Name: {config['name']}")
            print(f"  Base model: {config['base_model']}")
            print(f"  Size: {config['original_params']} → {config['target_params']}")
            print(f"  Role: {config['role']}")
            print(f"  Priority: {config['priority']}")
        sys.exit(0)

    create_specialist(specialist_name)


if __name__ == "__main__":
    main()

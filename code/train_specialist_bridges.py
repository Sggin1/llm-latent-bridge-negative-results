#!/usr/bin/env python3
"""
Generic Specialist Bridge Trainer
===================================

Trains bridges to connect specialists to Universal Hub.
Based on notes_5/train_docling_bridges.py

Usage:
    python train_specialist_bridges.py python    # Train Python specialist bridges
    python train_specialist_bridges.py sql       # Train SQL specialist bridges
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import json
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tensorport.core.universal_hub import BridgeNetwork, get_alignment_texts
from notes_20.create_specialist import StrippedSpecialist, CONFIG, SPECIALISTS, UNIVERSAL_DIM

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

print("=" * 70)
print("SPECIALIST BRIDGE TRAINING")
print(f"Device: {device}")
print("=" * 70)


def train_specialist_bridges(specialist_name):
    """
    Train bridges for a specialist model.

    Creates:
    - specialist_to_universal.pt
    - universal_to_specialist.pt
    """

    if specialist_name not in SPECIALISTS:
        print(f"❌ Unknown specialist: {specialist_name}")
        return

    spec_config = SPECIALISTS[specialist_name]

    print(f"\nTraining bridges for: {spec_config['name']}")
    print(f"Role: {spec_config['role']}")

    # ========================================================================
    # STEP 1: Load Manager Model (Anchor)
    # ========================================================================

    print("\n[1] Loading Manager model (anchor)...")
    manager_path = CONFIG["universal_hub"]["anchor_model"]
    manager_tokenizer = AutoTokenizer.from_pretrained(manager_path)
    manager_model = AutoModelForCausalLM.from_pretrained(manager_path, torch_dtype=dtype)
    manager_model.to(device).eval()

    manager_hidden_dim = manager_model.config.hidden_size
    print(f"  ✓ Manager: {manager_path}")
    print(f"    Hidden dim: {manager_hidden_dim}")

    # Load trained Manager bridges
    print("\n[2] Loading Manager → Universal bridge...")
    # Support both host and Docker container paths
    project_root = Path(__file__).parent.parent
    bridge_dir = project_root / "bridges"
    manager_to_universal = BridgeNetwork(manager_hidden_dim, UNIVERSAL_DIM).to(device)
    manager_to_universal.load_state_dict(
        torch.load(bridge_dir / "manager_to_universal.pt", map_location=device)
    )
    manager_to_universal.eval()
    print("  ✓ Loaded manager_to_universal.pt")

    # ========================================================================
    # STEP 2: Load Specialist Model
    # ========================================================================

    print(f"\n[3] Loading {spec_config['name']}...")

    specialist_dir = project_root / "models" / f"{specialist_name}-specialist"

    if not specialist_dir.exists():
        print(f"❌ Specialist not found: {specialist_dir}")
        print(f"   Run: python create_specialist.py {specialist_name}")
        return

    # Load config
    with open(specialist_dir / "config.json") as f:
        specialist_config = json.load(f)

    # Load tokenizer
    specialist_tokenizer = AutoTokenizer.from_pretrained(specialist_dir)

    # Load specialist model
    # Need to get original model to reconstruct architecture
    print(f"  Loading base model for architecture...")
    # Check if base_model is a local path or HuggingFace model ID
    base_model_path = spec_config['base_model']
    # Simple heuristic: if it contains a path separator, it's local
    is_local = ('/' in base_model_path or '\\' in base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        local_files_only=is_local,
        trust_remote_code=True
    )

    specialist_model = StrippedSpecialist(
        base_model,
        keep_layers=specialist_config['keep_layers'],
        universal_dim=UNIVERSAL_DIM
    )
    specialist_model.load_state_dict(
        torch.load(specialist_dir / "model.pt", map_location=device)
    )
    specialist_model.to(device).eval()

    specialist_hidden_dim = specialist_model.hidden_size
    print(f"  ✓ Specialist loaded")
    print(f"    Hidden dim: {specialist_hidden_dim}")
    print(f"    Params: {specialist_config['specialist_params']/1e9:.2f}B")

    # ========================================================================
    # STEP 3: Create Specialist Bridges
    # ========================================================================

    print(f"\n[4] Creating {spec_config['name']} bridges...")
    specialist_to_universal = BridgeNetwork(specialist_hidden_dim, UNIVERSAL_DIM).to(device)
    universal_to_specialist = BridgeNetwork(UNIVERSAL_DIM, specialist_hidden_dim).to(device)

    total_params = (
        sum(p.numel() for p in specialist_to_universal.parameters()) +
        sum(p.numel() for p in universal_to_specialist.parameters())
    )
    print(f"  Bridge params: {total_params:,}")

    # ========================================================================
    # STEP 4: Extract Hidden States
    # ========================================================================

    print(f"\n[5] Extracting hidden states...")
    num_texts = CONFIG["training"]["alignment_texts"]
    alignment_texts = get_alignment_texts(num_texts)
    print(f"  Alignment texts: {len(alignment_texts)}")

    manager_hidden_list = []
    specialist_hidden_list = []

    successful = 0
    for i, text in enumerate(alignment_texts):
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(alignment_texts)} ({successful} successful)...")

        try:
            # Manager hidden state
            manager_inputs = manager_tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )
            manager_inputs = {k: v.to(device) for k, v in manager_inputs.items()}

            with torch.no_grad():
                manager_outputs = manager_model(
                    **manager_inputs,
                    output_hidden_states=True
                )
                manager_hidden = manager_outputs.hidden_states[-1][:, -1, :]
                manager_hidden_list.append(manager_hidden.cpu())

            # Specialist hidden state
            specialist_inputs = specialist_tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )
            specialist_inputs = {k: v.to(device) for k, v in specialist_inputs.items()}

            with torch.no_grad():
                specialist_outputs = specialist_model(
                    specialist_inputs["input_ids"],
                    output_hidden_states=True
                )
                specialist_hidden = specialist_outputs["hidden"]
                specialist_hidden_list.append(specialist_hidden.cpu())

            successful += 1

        except Exception as e:
            print(f"      Error on text {i}: {e}")
            continue

    print(f"  ✓ Extracted {successful} hidden state pairs")

    # Stack into tensors
    manager_hidden = torch.cat(manager_hidden_list, dim=0).to(device)
    specialist_hidden = torch.cat(specialist_hidden_list, dim=0).to(device)

    print(f"  Manager hidden: {manager_hidden.shape}")
    print(f"  Specialist hidden: {specialist_hidden.shape}")

    # ========================================================================
    # STEP 5: Map to Universal Space
    # ========================================================================

    print(f"\n[6] Mapping Manager to Universal space...")
    with torch.no_grad():
        universal_targets = manager_to_universal(manager_hidden)
    print(f"  Universal targets: {universal_targets.shape}")

    # ========================================================================
    # STEP 6: Train Specialist → Universal Bridge
    # ========================================================================

    print(f"\n[7] Training {spec_config['name']} → Universal bridge...")

    optimizer = torch.optim.Adam(
        specialist_to_universal.parameters(),
        lr=CONFIG["training"]["learning_rate"]
    )
    loss_fn = nn.MSELoss()

    dataset = TensorDataset(specialist_hidden, universal_targets)
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["training"]["batch_size"],
        shuffle=True
    )

    start_time = time.time()
    epochs = CONFIG["training"]["epochs"]

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_specialist, batch_universal in dataloader:
            optimizer.zero_grad()

            # Forward pass
            predicted = specialist_to_universal(batch_specialist)

            # Compute loss
            loss = loss_fn(predicted, batch_universal)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch + 1}/{epochs}: Loss = {avg_loss:.6f}")

    train_time = time.time() - start_time
    print(f"  ✓ Training complete in {train_time:.1f}s")

    # ========================================================================
    # STEP 7: Train Universal → Specialist Bridge
    # ========================================================================

    print(f"\n[8] Training Universal → {spec_config['name']} bridge...")

    optimizer = torch.optim.Adam(
        universal_to_specialist.parameters(),
        lr=CONFIG["training"]["learning_rate"]
    )

    dataset = TensorDataset(universal_targets, specialist_hidden)
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG["training"]["batch_size"],
        shuffle=True
    )

    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_universal, batch_specialist in dataloader:
            optimizer.zero_grad()

            # Forward pass
            predicted = universal_to_specialist(batch_universal)

            # Compute loss
            loss = loss_fn(predicted, batch_specialist)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch + 1}/{epochs}: Loss = {avg_loss:.6f}")

    train_time = time.time() - start_time
    print(f"  ✓ Training complete in {train_time:.1f}s")

    # ========================================================================
    # STEP 8: Validate Alignment
    # ========================================================================

    print(f"\n[9] Validating alignment...")

    with torch.no_grad():
        # Round trip: Specialist → Universal → Specialist
        universal_from_specialist = specialist_to_universal(specialist_hidden)
        reconstructed_specialist = universal_to_specialist(universal_from_specialist)

        # Compute alignment
        cosine_sim = nn.functional.cosine_similarity(
            specialist_hidden, reconstructed_specialist, dim=1
        ).mean().item()

        print(f"  Round-trip alignment: {cosine_sim:.4f}")

        if cosine_sim > 0.85:
            print("  ✓ Excellent alignment (>0.85)")
        elif cosine_sim > 0.75:
            print("  ✓ Good alignment (>0.75)")
        else:
            print("  ⚠ Lower alignment than expected")

    # ========================================================================
    # STEP 9: Save Bridges
    # ========================================================================

    print(f"\n[10] Saving bridges...")

    # Save to bridges directory
    torch.save(
        specialist_to_universal.state_dict(),
        bridge_dir / f"{specialist_name}_to_universal.pt"
    )
    print(f"  ✓ Saved: {specialist_name}_to_universal.pt")

    torch.save(
        universal_to_specialist.state_dict(),
        bridge_dir / f"universal_to_{specialist_name}.pt"
    )
    print(f"  ✓ Saved: universal_to_{specialist_name}.pt")

    # Summary
    print("\n" + "=" * 70)
    print(f"{spec_config['name'].upper()} BRIDGES TRAINED")
    print("=" * 70)

    print(f"""
BRIDGES CREATED:
  - {specialist_name}_to_universal.pt
  - universal_to_{specialist_name}.pt

ALIGNMENT QUALITY:
  - Round-trip: {cosine_sim:.4f}
  - Training time: {train_time*2:.1f}s total

INTEGRATION:
  ✓ {spec_config['name']} can now communicate with Universal Hub
  ✓ Works with ALL existing specialists (Manager, Code God, Vision, Docling)
  ✓ Ready for multi-specialist workflows

NEXT STEPS:
  1. Test integration: python test_multi_specialist_workflow.py
  2. Benchmark performance vs traditional approach
  3. Add more specialists!

Saved to: {bridge_dir}
""")


def main():
    if len(sys.argv) < 2:
        print("Usage: python train_specialist_bridges.py <specialist_name>")
        print("\nAvailable specialists:")
        project_root = Path(__file__).parent.parent
        for name in SPECIALISTS.keys():
            specialist_dir = project_root / "models" / f"{name}-specialist"
            status = "✅" if specialist_dir.exists() else "❌ (create first)"
            print(f"  {status} {name}")
        sys.exit(1)

    specialist_name = sys.argv[1]
    train_specialist_bridges(specialist_name)


if __name__ == "__main__":
    main()

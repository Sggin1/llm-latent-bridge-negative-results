"""
Bridge Trainer
==============

Train neural bridges between model hidden spaces.

The bridge learns to transform hidden states from one model's space
to another's, enabling direct tensor-to-tensor communication between
different models (even different architectures/families).

Key insight: The bridge is tiny (~1-2M params) while models are frozen.
Training is fast because we only backprop through the bridge.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Optional, List, Tuple, Callable
from pathlib import Path
import time


class LinearBridge(nn.Module):
    """Simple linear projection bridge."""

    def __init__(self, source_dim: int, target_dim: int):
        super().__init__()
        self.proj = nn.Linear(source_dim, target_dim)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class MLPBridge(nn.Module):
    """MLP bridge with bottleneck for better generalization."""

    def __init__(
        self,
        source_dim: int,
        target_dim: int,
        bottleneck_dim: Optional[int] = None,
        num_layers: int = 2
    ):
        super().__init__()

        if bottleneck_dim is None:
            bottleneck_dim = min(source_dim, target_dim) // 2

        layers = []

        # Input layer
        layers.append(nn.Linear(source_dim, bottleneck_dim))
        layers.append(nn.GELU())

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(bottleneck_dim, bottleneck_dim))
            layers.append(nn.GELU())

        # Output layer
        layers.append(nn.Linear(bottleneck_dim, target_dim))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class CachedHiddenStatesDataset(Dataset):
    """Dataset of pre-computed hidden state pairs."""

    def __init__(
        self,
        source_hidden: torch.Tensor,
        target_hidden: torch.Tensor
    ):
        """
        Args:
            source_hidden: [num_samples, seq_len, source_dim] or [num_samples, source_dim]
            target_hidden: [num_samples, seq_len, target_dim] or [num_samples, target_dim]
        """
        assert len(source_hidden) == len(target_hidden)
        self.source = source_hidden
        self.target = target_hidden

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        return self.source[idx], self.target[idx]


class BridgeTrainer:
    """
    Train a neural bridge between two model hidden spaces.

    Usage:
        trainer = BridgeTrainer(source_model, target_model)
        trainer.cache_hidden_states(texts)  # Pre-compute (slow, once)
        trainer.train(epochs=50)            # Train bridge (fast)
        trainer.save("bridge.pt")
    """

    def __init__(
        self,
        source_model,
        target_model,
        source_tokenizer,
        target_tokenizer,
        bridge_type: str = "mlp",
        source_dim: Optional[int] = None,
        target_dim: Optional[int] = None,
        device: str = "cpu"
    ):
        self.source_model = source_model
        self.target_model = target_model
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.device = device

        # Get dimensions from model configs
        self.source_dim = source_dim or source_model.config.hidden_size
        self.target_dim = target_dim or target_model.config.hidden_size

        # Create bridge
        if bridge_type == "linear":
            self.bridge = LinearBridge(self.source_dim, self.target_dim)
        elif bridge_type == "mlp":
            self.bridge = MLPBridge(self.source_dim, self.target_dim)
        else:
            raise ValueError(f"Unknown bridge type: {bridge_type}")

        self.bridge.to(device)

        # Freeze models
        for param in source_model.parameters():
            param.requires_grad = False
        for param in target_model.parameters():
            param.requires_grad = False

        # Cache storage
        self.cached_source: Optional[torch.Tensor] = None
        self.cached_target: Optional[torch.Tensor] = None

        print(f"BridgeTrainer initialized:")
        print(f"  Source dim: {self.source_dim}")
        print(f"  Target dim: {self.target_dim}")
        print(f"  Bridge type: {bridge_type}")
        print(f"  Bridge params: {self.bridge.num_parameters:,}")

    def extract_hidden_states(
        self,
        model,
        tokenizer,
        texts: List[str],
        layer: int = -1,
        pool: str = "last"
    ) -> torch.Tensor:
        """
        Extract hidden states from a model for a list of texts.

        Args:
            model: The model to extract from
            tokenizer: Tokenizer for the model
            texts: List of input texts
            layer: Which layer to extract (-1 = last)
            pool: How to pool sequence ("last", "mean", "all")

        Returns:
            hidden_states: [num_texts, hidden_dim] or [num_texts, seq_len, hidden_dim]
        """
        model.eval()
        all_hidden = []

        for i, text in enumerate(texts):
            if (i + 1) % 100 == 0:
                print(f"    Processing {i + 1}/{len(texts)}...")

            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            hidden = outputs.hidden_states[layer]  # [1, seq_len, hidden]

            if pool == "last":
                hidden = hidden[:, -1, :]  # [1, hidden]
            elif pool == "mean":
                hidden = hidden.mean(dim=1)  # [1, hidden]
            elif pool == "all":
                pass  # Keep [1, seq_len, hidden]

            all_hidden.append(hidden.cpu())

        return torch.cat(all_hidden, dim=0)

    def cache_hidden_states(
        self,
        texts: List[str],
        pool: str = "last"
    ):
        """
        Pre-compute and cache hidden states for training.

        This is the slow step (model inference). Run once, then
        train many epochs on cached data.
        """
        print(f"\nCaching hidden states for {len(texts)} texts...")
        print(f"  Pooling method: {pool}")

        start = time.time()

        print("\n  Extracting from source model...")
        self.cached_source = self.extract_hidden_states(
            self.source_model,
            self.source_tokenizer,
            texts,
            pool=pool
        )

        print("\n  Extracting from target model...")
        self.cached_target = self.extract_hidden_states(
            self.target_model,
            self.target_tokenizer,
            texts,
            pool=pool
        )

        elapsed = time.time() - start
        print(f"\n  Caching complete in {elapsed:.1f}s")
        print(f"  Source shape: {self.cached_source.shape}")
        print(f"  Target shape: {self.cached_target.shape}")

    def train(
        self,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        log_every: int = 10
    ) -> List[float]:
        """
        Train the bridge on cached hidden states.

        Returns:
            List of loss values per epoch
        """
        if self.cached_source is None:
            raise RuntimeError("Must call cache_hidden_states() first!")

        dataset = CachedHiddenStatesDataset(
            self.cached_source,
            self.cached_target
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(
            self.bridge.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )

        criterion = nn.MSELoss()

        print(f"\nTraining bridge for {epochs} epochs...")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {lr}")
        print(f"  Batches per epoch: {len(loader)}")

        losses = []
        start = time.time()

        for epoch in range(epochs):
            epoch_loss = 0.0

            for source_batch, target_batch in loader:
                source_batch = source_batch.to(self.device)
                target_batch = target_batch.to(self.device)

                optimizer.zero_grad()

                predicted = self.bridge(source_batch)
                loss = criterion(predicted, target_batch)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            losses.append(avg_loss)
            scheduler.step()

            if (epoch + 1) % log_every == 0 or epoch == 0:
                elapsed = time.time() - start
                print(f"  Epoch {epoch + 1:3d}/{epochs}: loss={avg_loss:.6f} ({elapsed:.1f}s)")

        print(f"\nTraining complete!")
        print(f"  Final loss: {losses[-1]:.6f}")
        print(f"  Total time: {time.time() - start:.1f}s")

        return losses

    def save(self, path: str):
        """Save trained bridge to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "bridge_state": self.bridge.state_dict(),
            "source_dim": self.source_dim,
            "target_dim": self.target_dim,
            "bridge_type": type(self.bridge).__name__,
        }, path)

        print(f"Bridge saved to {path}")

    def load(self, path: str):
        """Load trained bridge from file."""
        data = torch.load(path, map_location=self.device)
        self.bridge.load_state_dict(data["bridge_state"])
        print(f"Bridge loaded from {path}")

    def translate(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Translate hidden states using the trained bridge."""
        self.bridge.eval()
        with torch.no_grad():
            return self.bridge(hidden_states.to(self.device))


def get_training_texts(num_samples: int = 1000) -> List[str]:
    """
    Get diverse training texts.

    For a real system, use a larger dataset like:
    - Wikipedia paragraphs
    - Common Crawl samples
    - Books/articles
    """
    # Simple diverse examples for quick testing
    templates = [
        "The capital of {} is a beautiful city.",
        "Machine learning is used for {}.",
        "In Python, you can {} using libraries.",
        "The weather today is {} and pleasant.",
        "Scientists discovered that {} is important.",
        "The best way to learn {} is practice.",
        "Technology has changed how we {}.",
        "Many people enjoy {} in their free time.",
        "The history of {} is fascinating.",
        "Research shows that {} improves health.",
    ]

    topics = [
        "France", "Germany", "Japan", "Brazil", "Australia",
        "programming", "cooking", "music", "sports", "art",
        "physics", "biology", "chemistry", "mathematics", "economics",
        "reading", "traveling", "photography", "gardening", "writing",
        "computers", "medicine", "education", "environment", "space",
    ]

    texts = []
    for i in range(num_samples):
        template = templates[i % len(templates)]
        topic = topics[i % len(topics)]
        texts.append(template.format(topic))

    return texts

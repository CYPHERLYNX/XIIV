"""
Streaming SGBD Orchestrator
============================
Implements Coreset-Anchored Streaming for training XIIV-1B on 80B+ tokens
within 50GB disk, without catastrophic forgetting.

Pipeline per dataset:
1. DOWNLOAD: Fetch dataset (up to 10B tokens)
2. TRAIN: SGBD block training on full dataset
3. CORESET: Extract 200M most informative tokens via gradient-based selection
4. FISHER: Compute compressed Fisher diagonal for EWC protection
5. DISCARD: Delete bulk dataset, keep coreset + Fisher
6. NEXT: Move to next dataset, replaying all previous coresets

Memory budget:
- Active dataset: ~20GB max
- Coresets (8 × 50M tokens): ~3GB
- Fisher diagonals (compressed): ~2GB
- Model + cache: ~20GB
- Total peak: ~45GB < 50GB ✓

Anti-forgetting (3 layers):
1. Smart coresets (gradient-selected, not random)
2. Progressive replay (30% old coresets in each batch)
3. EWC regularization (Fisher-weighted weight protection)
"""

import os
import sys
import torch
import json
import shutil
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict


@dataclass
class DatasetConfig:
    name: str
    source: str  # build_frontier_data.py source name
    max_tokens: int
    out_dir: str
    coreset_tokens: int = 200_000_000  # 200M tokens per coreset
    train_epochs: int = 1  # training passes over this dataset
    replay_boost: float = 1.0  # multiplier for this dataset's coreset in replay mix


# Streaming order: reasoning first (ASI driver), then code, then the rest
# Reasoning gets 3x training epochs + 2.5x replay weight: the model sees
# reasoning data far more than any other domain during training.
STREAMING_PLAN = [
    DatasetConfig("fineweb_edu", "fineweb-edu", 10_000_000_000, "data_frontier/fineweb_edu"),
    DatasetConfig("openthoughts", "openthoughts", 1_200_000_000, "data_frontier/reasoning",
                  train_epochs=3, replay_boost=2.5),
    # stackv2 DEFERRED 2026-10-15: gated HF dataset (needs auth + terms accepted);
    # content mapping also broken. Revisit separately; queue continues.
    DatasetConfig("papers", "pes2o", 10_000_000_000, "data_frontier/papers"),
    DatasetConfig("megamath", "megamath", 10_000_000_000, "data_frontier/megamath"),
    DatasetConfig("dclm", "dclm", 10_000_000_000, "data_frontier/dclm"),
    DatasetConfig("stackexchange", "stackexchange", 10_000_000_000, "data_frontier/qa"),
    DatasetConfig("pg19", "pg19", 5_000_000_000, "data_frontier/books"),
]


class StreamingSGBD:
    def __init__(
        self,
        base_dir: str,
        coreset_dir: str = "coresets",
        fisher_dir: str = "fisher",
        state_path: str = "streaming_state.json",
    ):
        self.base_dir = base_dir
        self.coreset_dir = os.path.join(base_dir, coreset_dir)
        self.fisher_dir = os.path.join(base_dir, fisher_dir)
        self.state_path = os.path.join(base_dir, state_path)
        
        os.makedirs(self.coreset_dir, exist_ok=True)
        os.makedirs(self.fisher_dir, exist_ok=True)
        
        self.state = self._load_state()
    
    def _load_state(self) -> Dict:
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                return json.load(f)
        return {
            "completed_datasets": [],
            "current_dataset": None,
            "total_tokens_streamed": 0,
        }
    
    def _save_state(self):
        with open(self.state_path, "w") as f:
            json.dump(self.state, f, indent=2)
    
    def get_completed_coresets(self) -> List[str]:
        """Get paths to all completed coresets for replay."""
        return [
            os.path.join(self.coreset_dir, f"{name}.pt")
            for name in self.state["completed_datasets"]
            if os.path.exists(os.path.join(self.coreset_dir, f"{name}.pt"))
        ]
    
    def get_completed_fishers(self) -> List[str]:
        """Get paths to all completed Fisher diagonals for EWC."""
        return [
            os.path.join(self.fisher_dir, f"{name}.pt")
            for name in self.state["completed_datasets"]
            if os.path.exists(os.path.join(self.fisher_dir, f"{name}.pt"))
        ]
    
    def mark_complete(
        self,
        dataset_name: str,
        tokens_streamed: int,
        coreset_path: str,
        fisher_path: str,
    ):
        """Mark a dataset as completed in streaming."""
        if dataset_name not in self.state["completed_datasets"]:
            self.state["completed_datasets"].append(dataset_name)
        self.state["total_tokens_streamed"] += tokens_streamed
        self.state["current_dataset"] = None
        self._save_state()
        print(f"[streaming] Completed {dataset_name}: "
              f"{tokens_streamed/1e9:.2f}B tokens streamed, "
              f"total: {self.state['total_tokens_streamed']/1e9:.2f}B")
    
    def discard_dataset(self, dataset_dir: str, keep_coreset: bool = True):
        """
        Discard bulk dataset after coreset extraction.
        Keeps only the coreset for replay.
        """
        full_path = os.path.join(self.base_dir, dataset_dir)
        if os.path.exists(full_path):
            # Calculate size before deletion
            size_gb = sum(
                os.path.getsize(os.path.join(dp, f)) / 1e9
                for dp, dn, fn in os.walk(full_path)
                for f in fn
            )
            shutil.rmtree(full_path)
            print(f"[streaming] Discarded {dataset_dir} ({size_gb:.1f}GB freed)")
    
    def get_replay_mix(
        self,
        current_dataset: str,
        replay_ratio: float = 0.3,
    ) -> Dict[str, float]:
        """
        Compute data mixing ratios for progressive replay.
        
        Returns dict mapping dataset name -> sampling probability.
        Current dataset gets (1 - replay_ratio), previous coresets
        split replay_ratio weighted by their replay_boost
        (reasoning data gets 2.5x — the ASI driver).
        """
        completed = [
            d for d in self.state["completed_datasets"]
            if d != current_dataset
        ]

        mix = {current_dataset: 1.0 - replay_ratio}

        if completed and replay_ratio > 0:
            # Weight by replay_boost (default 1.0, reasoning 2.5)
            plan_lookup = {d.name: d for d in STREAMING_PLAN}
            boosts = []
            for name in completed:
                cfg = plan_lookup.get(name)
                boosts.append(cfg.replay_boost if cfg else 1.0)
            total_boost = sum(boosts)
            for name, boost in zip(completed, boosts):
                mix[name] = replay_ratio * (boost / total_boost)

        return mix
    
    def print_status(self):
        """Print current streaming status."""
        print("\n" + "="*60)
        print("STREAMING SGBD STATUS")
        print("="*60)
        print(f"Completed: {len(self.state['completed_datasets'])}/{len(STREAMING_PLAN)} datasets")
        for name in self.state["completed_datasets"]:
            print(f"  ✓ {name}")
        
        remaining = [d.name for d in STREAMING_PLAN 
                     if d.name not in self.state["completed_datasets"]]
        if remaining:
            print(f"\nRemaining:")
            for name in remaining:
                print(f"  ○ {name}")
        
        print(f"\nTotal tokens streamed: {self.state['total_tokens_streamed']/1e9:.2f}B")
        
        # Disk usage
        coreset_size = sum(
            os.path.getsize(os.path.join(self.coreset_dir, f)) / 1e9
            for f in os.listdir(self.coreset_dir)
            if f.endswith(".pt")
        ) if os.path.exists(self.coreset_dir) else 0
        
        fisher_size = sum(
            os.path.getsize(os.path.join(self.fisher_dir, f)) / 1e9
            for f in os.listdir(self.fisher_dir)
            if f.endswith(".pt")
        ) if os.path.exists(self.fisher_dir) else 0
        
        print(f"Coresets: {coreset_size:.2f}GB, Fisher: {fisher_size:.2f}GB")
        print("="*60 + "\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Streaming SGBD orchestrator")
    parser.add_argument("--base-dir", default=".",
                        help="Base directory for streaming")
    parser.add_argument("--status", action="store_true",
                        help="Print status and exit")
    args = parser.parse_args()
    
    streamer = StreamingSGBD(args.base_dir)
    
    if args.status:
        streamer.print_status()
    else:
        streamer.print_status()
        print("\nStreaming plan:")
        for i, ds in enumerate(STREAMING_PLAN, 1):
            status = "✓" if ds.name in streamer.state["completed_datasets"] else "○"
            print(f"  {status} {i}. {ds.name}: {ds.max_tokens/1e9:.0f}B tokens")


if __name__ == "__main__":
    main()

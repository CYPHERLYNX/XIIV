"""
Benchmark Contamination Detection
=================================
13-gram overlap checking against eval benchmark sets.

Standard methodology (LLaMA-3, GPT-4, lm_eval):
- Extract all 13-grams from benchmark questions
- Hash into a set (memory-light, benchmarks are tiny)
- Filter training docs: any doc containing a benchmark 13-gram = contaminated
- Remove contaminated docs from training

This is what separates honest benchmark numbers from memorization artifacts.
"""

import hashlib
from typing import Set, List, Dict


def extract_ngrams(text: str, n: int = 13) -> Set[str]:
    """Extract word n-grams from text."""
    words = text.lower().split()
    if len(words) < n:
        return set()
    return {" ".join(words[i:i+n]) for i in range(len(words) - n + 1)}


def hash_ngram(ngram: str) -> str:
    """Hash n-gram for compact storage."""
    return hashlib.md5(ngram.encode()).hexdigest()[:16]


class ContaminationFilter:
    """
    Benchmark decontamination filter.
    
    Build once from benchmark files, then check every training doc.
    """
    
    def __init__(self, n: int = 13):
        self.n = n
        self.benchmark_ngrams: Set[str] = set()
        self.benchmark_sources: Dict[str, int] = {}  # source -> ngram count
        self.stats = {"checked": 0, "contaminated": 0, "clean": 0}
    
    def add_benchmark_file(self, path: str, source_name: str):
        """
        Add a benchmark file to the contamination set.
        Each line is treated as a separate benchmark item.
        """
        count_before = len(self.benchmark_ngrams)
        
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    for ng in extract_ngrams(line, self.n):
                        self.benchmark_ngrams.add(hash_ngram(ng))
        except FileNotFoundError:
            print(f"[decontam] Warning: benchmark file not found: {path}")
            return
        
        added = len(self.benchmark_ngrams) - count_before
        self.benchmark_sources[source_name] = added
        print(f"[decontam] Added {added} n-grams from {source_name} "
              f"(total: {len(self.benchmark_ngrams)})")
    
    def is_contaminated(self, text: str) -> bool:
        """
        Check if text contains any benchmark n-gram.
        Returns True if contaminated (should reject).
        """
        self.stats["checked"] += 1
        
        for ng in extract_ngrams(text, self.n):
            if hash_ngram(ng) in self.benchmark_ngrams:
                self.stats["contaminated"] += 1
                return True
        
        self.stats["clean"] += 1
        return False
    
    def get_stats(self) -> Dict:
        """Get contamination statistics."""
        checked = self.stats["checked"]
        return {
            **self.stats,
            "contamination_rate": self.stats["contaminated"] / max(checked, 1),
            "benchmark_ngrams": len(self.benchmark_ngrams),
            "sources": dict(self.benchmark_sources),
        }
    
    def memory_mb(self) -> float:
        """Estimate memory usage."""
        # 16 bytes per hash + Python set overhead ~72 bytes per entry
        # Rough estimate: ~100 bytes per n-gram
        return len(self.benchmark_ngrams) * 100 / 1e6

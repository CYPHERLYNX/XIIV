"""
Per-Domain fastText Quality Filter
==================================
Trains binary fastText classifiers to distinguish high-quality
reference text from random web text, per domain.

Based on DataComp-LM findings: fastText with the RIGHT reference set
beats more elaborate filtering approaches. Trains and scores in
milliseconds per doc on CPU.

Usage:
    1. Train: python -m xiiv.quality_filter --train --domain code \
                  --positive data/curated/code.txt \
                  --negative data/random/web.txt
    2. Score: python -m xiiv.quality_filter --score --domain code \
                  --input docs.txt --threshold 0.7

If fastText is not installed: pip install fasttext-wheel
"""

import os
import sys
import argparse


# Domain-specific positive reference descriptions
# (actual training data paths resolved at runtime)
DOMAIN_REFERENCES = {
    "code": {
        "description": "High-quality code: well-documented, tested, idiomatic",
        "positive_hint": "Use curated code from Stack v2 filtered by stars/tests",
        "negative_hint": "Random web text, non-code documents",
    },
    "papers": {
        "description": "High-quality scientific papers: clear methodology, results",
        "positive_hint": "Use peS2o papers with high citation counts",
        "negative_hint": "Random web text, low-quality preprints",
    },
    "math": {
        "description": "High-quality math: correct derivations, clear steps",
        "positive_hint": "Use MegaMath verified correct solutions",
        "negative_hint": "Random web text, incorrect math",
    },
    "reasoning": {
        "description": "High-quality reasoning: valid chain-of-thought",
        "positive_hint": "Use OpenThoughts3 verified correct traces",
        "negative_hint": "Random web text, incoherent reasoning",
    },
    "qa": {
        "description": "High-quality Q&A: accepted answers, clear explanations",
        "positive_hint": "Use StackExchange accepted/high-score answers",
        "negative_hint": "Random web text, unanswered questions",
    },
    "books": {
        "description": "High-quality books: published, edited prose",
        "positive_hint": "Use PG-19 published books",
        "negative_hint": "Random web text",
    },
    "web": {
        "description": "High-quality web: educational, factual, well-written",
        "positive_hint": "Use FineWeb-Edu high-score documents",
        "negative_hint": "Random unfiltered web text",
    },
}


def check_fasttext():
    """Check if fastText is available."""
    try:
        import fasttext
        return True
    except ImportError:
        return False


def prepare_training_file(positive_path, negative_path, output_path, max_per_class=100000):
    """
    Prepare fastText training file in the required format:
    __label__positive <text>
    __label__negative <text>
    """
    print(f"[quality] Preparing training file: {output_path}")
    
    count_pos = 0
    count_neg = 0
    
    with open(output_path, "w", encoding="utf-8") as out:
        # Positive examples
        if os.path.exists(positive_path):
            with open(positive_path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if count_pos >= max_per_class:
                        break
                    text = line.strip().replace("\n", " ")
                    if len(text) > 50:  # Skip tiny fragments
                        out.write(f"__label__positive {text}\n")
                        count_pos += 1
        
        # Negative examples
        if os.path.exists(negative_path):
            with open(negative_path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if count_neg >= max_per_class:
                        break
                    text = line.strip().replace("\n", " ")
                    if len(text) > 50:
                        out.write(f"__label__negative {text}\n")
                        count_neg += 1
    
    print(f"[quality] Wrote {count_pos} positive + {count_neg} negative examples")
    return count_pos, count_neg


def train_classifier(train_path, model_path, dim=128, epoch=10, lr=0.5):
    """Train fastText classifier."""
    import fasttext
    
    print(f"[quality] Training classifier: {model_path}")
    print(f"[quality] dim={dim} epoch={epoch} lr={lr}")
    
    model = fasttext.train_supervised(
        train_path,
        dim=dim,
        epoch=epoch,
        lr=lr,
        wordNgrams=2,
        loss="ova",
        verbose=1,
    )
    
    model.save_model(model_path)
    print(f"[quality] Saved to {model_path}")
    
    # Quick eval on training data
    result = model.test(train_path)
    print(f"[quality] Train precision: {result[1]:.3f}, recall: {result[2]:.3f}")
    
    return model


def score_documents(model_path, input_texts, threshold=0.7):
    """
    Score documents, return (text, score, keep) tuples.
    
    Args:
        model_path: Path to trained fastText model
        input_texts: List of document strings
        threshold: Minimum positive-class probability to keep
    
    Returns:
        List of (text, score, keep) tuples
    """
    import fasttext
    import numpy as np
    
    model = fasttext.load_model(model_path)
    results = []
    
    # Batch prediction for speed
    # fastText predict is fast but we batch for efficiency
    batch_size = 1000
    for i in range(0, len(input_texts), batch_size):
        batch = input_texts[i:i + batch_size]
        # Clean texts
        batch_clean = [t.replace("\n", " ")[:10000] for t in batch]
        
        labels, probs = model.predict(batch_clean, k=2)
        
        for text, lbls, prbs in zip(batch, labels, probs):
            # Find positive class probability
            score = 0.0
            for lbl, prb in zip(lbls, prbs):
                if "positive" in lbl:
                    score = float(prb)
                    break
            
            keep = score >= threshold
            results.append((text, score, keep))
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Per-domain fastText quality filter")
    parser.add_argument("--train", action="store_true", help="Train classifier")
    parser.add_argument("--score", action="store_true", help="Score documents")
    parser.add_argument("--domain", required=True,
                        choices=list(DOMAIN_REFERENCES.keys()),
                        help="Domain to train/score for")
    parser.add_argument("--positive", help="Path to positive examples (one doc per line)")
    parser.add_argument("--negative", help="Path to negative examples (one doc per line)")
    parser.add_argument("--model-dir", default="quality_models",
                        help="Directory for trained models")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Quality threshold for scoring")
    parser.add_argument("--input", help="Input file to score (one doc per line)")
    parser.add_argument("--output", help="Output file for kept docs")
    
    args = parser.parse_args()
    
    if not check_fasttext():
        print("[quality] ERROR: fastText not installed.")
        print("[quality] Install with: pip install fasttext-wheel")
        sys.exit(1)
    
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, f"{args.domain}.bin")
    
    if args.train:
        if not args.positive or not args.negative:
            print("[quality] ERROR: --positive and --negative required for training")
            print(f"[quality] Domain '{args.domain}': {DOMAIN_REFERENCES[args.domain]['description']}")
            print(f"[quality] Positive hint: {DOMAIN_REFERENCES[args.domain]['positive_hint']}")
            print(f"[quality] Negative hint: {DOMAIN_REFERENCES[args.domain]['negative_hint']}")
            sys.exit(1)
        
        train_path = os.path.join(args.model_dir, f"{args.domain}_train.txt")
        prepare_training_file(args.positive, args.negative, train_path)
        train_classifier(train_path, model_path)
    
    elif args.score:
        if not os.path.exists(model_path):
            print(f"[quality] ERROR: Model not found: {model_path}")
            print(f"[quality] Train first with --train --domain {args.domain}")
            sys.exit(1)
        
        if not args.input:
            print("[quality] ERROR: --input required for scoring")
            sys.exit(1)
        
        with open(args.input, encoding="utf-8", errors="ignore") as f:
            texts = [line.strip() for line in f if line.strip()]
        
        print(f"[quality] Scoring {len(texts)} documents...")
        results = score_documents(model_path, texts, args.threshold)
        
        kept = sum(1 for _, _, k in results if k)
        print(f"[quality] Kept {kept}/{len(results)} ({kept/len(results)*100:.1f}%)")
        
        if args.output:
            with open(args.output, "w", encoding="utf-8") as out:
                for text, score, keep in results:
                    if keep:
                        out.write(text + "\n")
            print(f"[quality] Wrote kept docs to {args.output}")


if __name__ == "__main__":
    main()

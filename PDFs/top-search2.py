#!/usr/bin/env python3
"""
Enhanced Dataset Quality Analyzer
==================================
Analyzes Q&A datasets with semantic themes and quality metrics.

Shows:
- Top themes by frequency
- Quality distribution
- Semantic diversity
- Sample high-quality entries
"""

import gzip
import json
import re
import collections
from pathlib import Path

def analyze_dataset(path):
    """Analyze Q&A dataset and return comprehensive stats"""
    
    print(f"📊 Analyzing: {path}\n")
    
    # Counters
    theme_counter = collections.Counter()
    quality_scores = []
    theme_diversity = collections.defaultdict(set)
    total_entries = 0
    
    # First pass: gather statistics
    print("🔍 Scanning dataset...")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                total_entries += 1
                
                # Extract themes
                themes = (obj.get("source_metadata", {}).get("semantic_themes") or 
                         obj.get("semantic_themes") or [])
                
                if isinstance(themes, list):
                    for t in themes:
                        if isinstance(t, str) and t.strip():
                            clean_theme = t.strip().lower()
                            theme_counter[clean_theme] += 1
                            # Track which files contributed to this theme
                            source = obj.get("source_metadata", {}).get("source_file", "unknown")
                            theme_diversity[clean_theme].add(source)
                
                # Extract quality score
                quality = obj.get("quality_metrics", {}).get("quality_score")
                if quality is not None:
                    quality_scores.append(float(quality))
                    
            except Exception as e:
                continue
    
    print(f"✓ Processed {total_entries:,} entries\n")
    
    # Calculate statistics
    top_themes = theme_counter.most_common(25)
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
    
    print("=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    print(f"Total Q&A pairs: {total_entries:,}")
    print(f"Unique themes: {len(theme_counter):,}")
    print(f"Average quality score: {avg_quality:.3f}")
    print(f"Quality range: {min(quality_scores):.3f} - {max(quality_scores):.3f}")
    
    # Show top themes with diversity metrics
    print("\n" + "=" * 70)
    print("TOP 25 SEMANTIC THEMES")
    print("=" * 70)
    print(f"{'Theme':<40} {'Count':>8} {'Sources':>8}")
    print("-" * 70)
    for theme, count in top_themes:
        num_sources = len(theme_diversity[theme])
        print(f"{theme:<40} {count:>8} {num_sources:>8}")
    
    # Find high-quality examples
    print("\n🔍 Finding high-quality examples from top themes...")
    top10_themes = [t[0] for t in top_themes[:10]]
    
    # Second pass: find high-quality matches
    matches = []
    pattern = re.compile("|".join(re.escape(t) for t in top10_themes), re.IGNORECASE)
    
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                if pattern.search(line):
                    entry = json.loads(line.strip())
                    quality = entry.get("quality_metrics", {}).get("quality_score", 0)
                    if quality >= 0.7:  # Only high-quality entries
                        matches.append((quality, entry))
            except Exception:
                continue
    
    # Sort by quality descending
    matches.sort(reverse=True, key=lambda x: x[0])
    
    print(f"✓ Found {len(matches):,} high-quality matches\n")
    
    # Show top examples
    print("=" * 70)
    print("HIGH-QUALITY EXAMPLES (Top 10)")
    print("=" * 70)
    
    for i, (quality, entry) in enumerate(matches[:10], 1):
        user = entry.get("user", "")[:200]
        assistant = entry.get("assistant", "")[:200]
        themes = entry.get("source_metadata", {}).get("semantic_themes", [])[:3]
        source = entry.get("source_metadata", {}).get("source_file", "unknown")
        
        print(f"\n[{i}] Quality: {quality:.3f} | Themes: {', '.join(themes)}")
        print(f"    Source: {source}")
        print(f"    Q: {user}...")
        print(f"    A: {assistant}...")
    
    # Return data for programmatic use
    return {
        "total_entries": total_entries,
        "unique_themes": len(theme_counter),
        "average_quality": avg_quality,
        "top_25_themes": [(t, c, len(theme_diversity[t])) for t, c in top_themes],
        "high_quality_count": len(matches),
        "sample_entries": [
            {
                "quality": q,
                "user": e.get("user", "")[:300],
                "assistant": e.get("assistant", "")[:300],
                "themes": e.get("source_metadata", {}).get("semantic_themes", [])[:3],
                "source": e.get("source_metadata", {}).get("source_file", "unknown")
            }
            for q, e in matches[:25]
        ]
    }

def compare_splits(prefix="dataset"):
    """Compare train/validation/test splits"""
    
    print("\n" + "=" * 70)
    print("COMPARING DATA SPLITS")
    print("=" * 70)
    
    for split in ["train", "validation", "test"]:
        path = f"{prefix}_qa_{split}.jsonl.gz"
        if Path(path).exists():
            print(f"\n📂 {split.upper()}:")
            
            # Quick stats
            count = 0
            themes = set()
            qualities = []
            
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        count += 1
                        
                        theme_list = (obj.get("source_metadata", {}).get("semantic_themes") or 
                                    obj.get("semantic_themes") or [])
                        themes.update(t.lower() for t in theme_list if isinstance(t, str))
                        
                        quality = obj.get("quality_metrics", {}).get("quality_score")
                        if quality:
                            qualities.append(float(quality))
                    except:
                        continue
            
            avg_q = sum(qualities) / len(qualities) if qualities else 0
            print(f"  Entries: {count:,}")
            print(f"  Unique themes: {len(themes):,}")
            print(f"  Avg quality: {avg_q:.3f}")

if __name__ == "__main__":
    import sys
    
    # Default path
    path = ""/mnt/data/Hofstadter.jsonl.gz""
    
    # Allow override from command line
    if len(sys.argv) > 1:
        path = sys.argv[1]
    
    # Main analysis
    results = analyze_dataset(path)
    
    # Compare splits if available
    compare_splits()
    
    print("\n" + "=" * 70)
    print("✓ Analysis complete!")
    print("=" * 70)

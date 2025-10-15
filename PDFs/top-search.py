import gzip
import json
import re
import collections

# Step 1: Recreate embedded_theme_top25 by scanning for semantic themes
path = "/mnt/data/dataset_qa_train.jsonl.gz"
theme_counter = collections.Counter()

with gzip.open(path, "rt", encoding="utf-8") as f:
    for line in f:
        try:
            obj = json.loads(line)
            themes = obj.get("source_metadata", {}).get("semantic_themes") or obj.get("semantic_themes")
            if isinstance(themes, list):
                for t in themes:
                    if isinstance(t, str) and t.strip():
                        theme_counter[t.strip().lower()] += 1
        except Exception:
            continue

embedded_theme_top25 = theme_counter.most_common(25)
top10 = [t[0] for t in embedded_theme_top25[:10]]

# Step 2: Build regex pattern
pattern = re.compile("|".join(re.escape(t) for t in top10), re.IGNORECASE) if top10 else re.compile("")

# Step 3: Search for those themes in the same file
matches = []
with gzip.open(path, "rt", encoding="utf-8") as f:
    for line in f:
        try:
            if pattern.search(line):
                entry = json.loads(line.strip())
                matches.append(entry)
        except Exception:
            continue

# Step 4: Build previews (sample of 25)
previews = []
for m in matches[:25]:
    previews.append({
        "user": (m.get("user") or "")[:300],
        "assistant": (m.get("assistant") or "")[:300],
        "quality": m.get("quality_metrics", {}).get("quality_score")
    })

{
    "top_10_themes": top10,
    "total_matches": len(matches),
    "sample_previews": previews
}


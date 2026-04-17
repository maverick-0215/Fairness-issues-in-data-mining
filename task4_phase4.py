import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def ensure_numpy() -> Any:
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError("NumPy is required for Phase 4. Install with: pip install numpy") from exc
    return np


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def cosine_similarity(np: Any, a: Any, b: Any) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def get_word_vectors(
    manifest: Dict[str, object],
    centroids: Any,
    category: str,
    set_name: str,
) -> List[Tuple[str, Any]]:
    items: List[Tuple[str, Any]] = []
    words = manifest["words"][category][set_name]
    for word, info in words.items():
        if not info.get("has_context", False):
            continue
        key = info["npz_key"]
        items.append((word, centroids[key]))
    return items


def compute_bias_rows(np: Any, manifest: Dict[str, object], centroids: Any) -> List[Dict[str, object]]:
    male_targets = get_word_vectors(manifest, centroids, "target", "male")
    female_targets = get_word_vectors(manifest, centroids, "target", "female")

    if not male_targets or not female_targets:
        raise RuntimeError("Target sets are empty after filtering for available context.")

    rows: List[Dict[str, object]] = []
    attribute_sets = manifest["words"]["attribute"]
    for attribute_set_name, words in attribute_sets.items():
        for attribute_word, info in words.items():
            if not info.get("has_context", False):
                continue
            attr_vec = centroids[info["npz_key"]]

            male_sims = [cosine_similarity(np, attr_vec, tvec) for _, tvec in male_targets]
            female_sims = [cosine_similarity(np, attr_vec, tvec) for _, tvec in female_targets]

            mean_male = float(np.mean(male_sims))
            mean_female = float(np.mean(female_sims))
            bias = mean_male - mean_female
            abs_bias = abs(bias)

            rows.append(
                {
                    "attribute_set": attribute_set_name,
                    "attribute_word": attribute_word,
                    "mean_cosine_to_male": mean_male,
                    "mean_cosine_to_female": mean_female,
                    "bias_score": bias,
                    "absolute_bias": abs_bias,
                    "neutrality_distance": abs_bias,
                }
            )
    return rows


def summarize_rows(np: Any, rows: List[Dict[str, object]]) -> Dict[str, object]:
    if not rows:
        raise RuntimeError("No attribute rows available for Phase 4 computation.")

    by_set: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_set.setdefault(row["attribute_set"], []).append(row)

    set_summaries: Dict[str, Dict[str, float]] = {}
    for set_name, set_rows in by_set.items():
        biases = np.array([r["bias_score"] for r in set_rows], dtype=np.float32)
        abs_biases = np.array([r["absolute_bias"] for r in set_rows], dtype=np.float32)
        set_summaries[set_name] = {
            "word_count": int(len(set_rows)),
            "mean_bias": float(np.mean(biases)),
            "mean_absolute_bias": float(np.mean(abs_biases)),
            "max_absolute_bias": float(np.max(abs_biases)),
        }

    all_abs = np.array([r["absolute_bias"] for r in rows], dtype=np.float32)
    top_rows = sorted(rows, key=lambda r: r["absolute_bias"], reverse=True)[:10]
    return {
        "total_attribute_words": int(len(rows)),
        "overall_mean_absolute_bias": float(np.mean(all_abs)),
        "overall_max_absolute_bias": float(np.max(all_abs)),
        "set_summaries": set_summaries,
        "top_biased_words": top_rows,
    }


def save_csv(rows: List[Dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "attribute_set",
        "attribute_word",
        "mean_cosine_to_male",
        "mean_cosine_to_female",
        "bias_score",
        "absolute_bias",
        "neutrality_distance",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(payload: Dict[str, object], output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4: fairness quantification using centroid cosine similarities."
    )
    parser.add_argument(
        "--input-centroids-file",
        type=Path,
        default=Path("outputs") / "task3_phase3_centroids.npz",
        help="Phase 3 centroid NPZ",
    )
    parser.add_argument(
        "--input-manifest-file",
        type=Path,
        default=Path("outputs") / "task3_phase3_manifest.json",
        help="Phase 3 centroid manifest JSON",
    )
    parser.add_argument(
        "--output-results-json",
        type=Path,
        default=Path("outputs") / "task4_phase4_results.json",
        help="Phase 4 detailed results JSON",
    )
    parser.add_argument(
        "--output-results-csv",
        type=Path,
        default=Path("outputs") / "task4_phase4_word_bias.csv",
        help="Phase 4 per-word results CSV",
    )
    args = parser.parse_args()

    try:
        np = ensure_numpy()
    except RuntimeError as exc:
        raise SystemExit(f"Dependency error: {exc}") from exc

    manifest = load_json(args.input_manifest_file)
    centroids = np.load(args.input_centroids_file, allow_pickle=False)

    rows = compute_bias_rows(np, manifest, centroids)
    summary = summarize_rows(np, rows)
    payload = {
        "phase": 4,
        "description": "Fairness quantification with Bias(a) = mean(cos(a, male)) - mean(cos(a, female))",
        "model_name": manifest.get("model_name"),
        "rows": rows,
        "summary": summary,
    }

    save_json(payload, args.output_results_json)
    save_csv(rows, args.output_results_csv)

    print(f"Wrote Phase 4 results to {args.output_results_json}")
    print(f"Wrote Phase 4 per-word CSV to {args.output_results_csv}")


if __name__ == "__main__":
    main()

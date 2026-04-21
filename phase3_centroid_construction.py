import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple


def ensure_numpy() -> Any:
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError("NumPy is required for Phase 3. Install with: pip install numpy") from exc
    return np


def load_phase2_manifest(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_phase2_embeddings(np: Any, path: Path) -> Any:
    return np.load(path, allow_pickle=False)


def centroid_for_word(np: Any, vectors: Any, hidden_size: int) -> Tuple[Any, int]:
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    if vectors.shape[0] == 0:
        return np.zeros((hidden_size,), dtype=np.float32), 0
    return vectors.mean(axis=0).astype(np.float32), int(vectors.shape[0])


def process_centroids(
    np: Any,
    phase2_manifest: Dict[str, object],
    phase2_embeddings: Any,
) -> Tuple[Dict[str, Any], Dict[str, object]]:
    hidden_size = int(phase2_manifest["hidden_size"])
    centroid_arrays: Dict[str, Any] = {}
    words_manifest: Dict[str, Dict[str, Dict[str, Dict[str, object]]]] = {}

    words = phase2_manifest["words"]
    for category, named_sets in words.items():
        words_manifest.setdefault(category, {})
        for set_name, set_words in named_sets.items():
            words_manifest[category].setdefault(set_name, {})
            for word, info in set_words.items():
                npz_key = info["npz_key"]
                vectors = phase2_embeddings[npz_key]
                centroid, source_count = centroid_for_word(np, vectors, hidden_size)
                centroid_key = npz_key
                centroid_arrays[centroid_key] = centroid

                words_manifest[category][set_name][word] = {
                    "npz_key": centroid_key,
                    "centroid_dim": int(centroid.shape[0]),
                    "source_vector_count": source_count,
                    "has_context": bool(source_count > 0),
                }

    output_manifest = {
        "phase": 3,
        "description": "Mean pooling and centroid calculation from Phase 2 contextual embeddings",
        "model_name": phase2_manifest.get("model_name"),
        "hidden_size": hidden_size,
        "words": words_manifest,
    }

    return centroid_arrays, output_manifest


def save_outputs(
    np: Any,
    centroids: Dict[str, Any],
    output_npz: Path,
    output_manifest_file: Path,
    output_manifest: Dict[str, object],
) -> None:
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_file.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(output_npz, **centroids)
    with output_manifest_file.open("w", encoding="utf-8") as f:
        json.dump(output_manifest, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3: mean-pool contextual vectors into one centroid per concept word."
    )
    parser.add_argument(
        "--input-embeddings-file",
        type=Path,
        default=Path("outputs") / "task2_phase2_embeddings.npz",
        help="Phase 2 embeddings .npz file",
    )
    parser.add_argument(
        "--input-manifest-file",
        type=Path,
        default=Path("outputs") / "task2_phase2_manifest.json",
        help="Phase 2 manifest .json file",
    )
    parser.add_argument(
        "--output-centroids-file",
        type=Path,
        default=Path("outputs") / "task3_phase3_centroids.npz",
        help="Output .npz file with one centroid vector per word",
    )
    parser.add_argument(
        "--output-manifest-file",
        type=Path,
        default=Path("outputs") / "task3_phase3_manifest.json",
        help="Output .json metadata file for centroid vectors",
    )
    args = parser.parse_args()

    try:
        np = ensure_numpy()
    except RuntimeError as exc:
        raise SystemExit(f"Dependency error: {exc}") from exc

    phase2_manifest = load_phase2_manifest(args.input_manifest_file)
    phase2_embeddings = load_phase2_embeddings(np, args.input_embeddings_file)
    centroids, output_manifest = process_centroids(np, phase2_manifest, phase2_embeddings)
    save_outputs(
        np=np,
        centroids=centroids,
        output_npz=args.output_centroids_file,
        output_manifest_file=args.output_manifest_file,
        output_manifest=output_manifest,
    )

    print(f"Wrote Phase 3 centroids to {args.output_centroids_file}")
    print(f"Wrote Phase 3 manifest to {args.output_manifest_file}")


if __name__ == "__main__":
    main()

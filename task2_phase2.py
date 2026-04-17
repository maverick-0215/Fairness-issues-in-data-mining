import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class WordAssignment:
    category: str
    set_name: str
    word: str
    source_file: str


def load_phase1_contexts(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compile_word_patterns(phase1_payload: Dict[str, object]) -> Dict[str, re.Pattern[str]]:
    contexts = phase1_payload["contexts"]
    patterns: Dict[str, re.Pattern[str]] = {}
    for category, named_sets in contexts.items():
        for set_name, words in named_sets.items():
            for word in words.keys():
                if word not in patterns:
                    patterns[word] = re.compile(
                        rf"(?<!\w){re.escape(word)}(?!\w)", re.IGNORECASE
                    )
    return patterns


def flatten_sentence_assignments(
    phase1_payload: Dict[str, object],
) -> Dict[str, Dict[str, object]]:
    sentence_map: Dict[str, Dict[str, object]] = {}
    contexts = phase1_payload["contexts"]

    for category, named_sets in contexts.items():
        for set_name, words in named_sets.items():
            for word, items in words.items():
                for item in items:
                    sentence = item["sentence"]
                    source_file = item["source_file"]
                    sentence_id = f"{source_file}::{sentence}"
                    if sentence_id not in sentence_map:
                        sentence_map[sentence_id] = {
                            "sentence": sentence,
                            "assignments": [],
                        }
                    sentence_map[sentence_id]["assignments"].append(
                        WordAssignment(
                            category=category,
                            set_name=set_name,
                            word=word,
                            source_file=source_file,
                        )
                    )

    return sentence_map


def find_word_spans(sentence: str, word_pattern: re.Pattern[str]) -> List[Tuple[int, int]]:
    return [match.span() for match in word_pattern.finditer(sentence)]


def overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return start_a < end_b and start_b < end_a


def ensure_dependencies() -> Tuple[object, object, object]:
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "NumPy is required for Phase 2. Install with: pip install numpy"
        ) from exc

    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for Phase 2. "
            "Install with a standard CPython environment and run: pip install torch transformers"
        ) from exc

    try:
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for Phase 2. Install with: pip install transformers"
        ) from exc

    return np, torch, (AutoTokenizer, AutoModel)


def extract_contextual_vectors(
    phase1_payload: Dict[str, object],
    model_name: str,
    device_preference: str,
) -> Tuple[Dict[Tuple[str, str, str], Any], Dict[Tuple[str, str, str], Dict[str, int]], int]:
    np, torch, (AutoTokenizer, AutoModel) = ensure_dependencies()
    sentence_map = flatten_sentence_assignments(phase1_payload)
    patterns = compile_word_patterns(phase1_payload)

    if device_preference == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_preference

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.to(device)
    model.eval()

    vectors_by_word: Dict[Tuple[str, str, str], List[Any]] = defaultdict(list)
    stats: Dict[Tuple[str, str, str], Dict[str, int]] = defaultdict(
        lambda: {"sentences_seen": 0, "occurrences_mapped": 0, "occurrences_unmapped": 0}
    )
    hidden_size = int(model.config.hidden_size)

    with torch.no_grad():
        for sentence_data in sentence_map.values():
            sentence = sentence_data["sentence"]
            assignments: List[WordAssignment] = sentence_data["assignments"]

            encoded = tokenizer(
                sentence,
                return_offsets_mapping=True,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            offset_mapping = encoded.pop("offset_mapping")[0].tolist()
            model_inputs = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**model_inputs)

            hidden_states = outputs.hidden_states
            token_embeddings = torch.stack(hidden_states[-4:], dim=0).mean(dim=0)[0]

            for assignment in assignments:
                word_key = (assignment.category, assignment.set_name, assignment.word)
                stats[word_key]["sentences_seen"] += 1

                spans = find_word_spans(sentence, patterns[assignment.word])
                for span_start, span_end in spans:
                    token_indices = []
                    for idx, (tok_start, tok_end) in enumerate(offset_mapping):
                        if tok_start == tok_end:
                            continue
                        if overlap(span_start, span_end, tok_start, tok_end):
                            token_indices.append(idx)

                    if not token_indices:
                        stats[word_key]["occurrences_unmapped"] += 1
                        continue

                    word_vec = token_embeddings[token_indices].mean(dim=0).cpu().numpy()
                    vectors_by_word[word_key].append(word_vec.astype(np.float32))
                    stats[word_key]["occurrences_mapped"] += 1

    dense_vectors: Dict[Tuple[str, str, str], Any] = {}
    for word_key, vectors in vectors_by_word.items():
        if vectors:
            dense_vectors[word_key] = np.stack(vectors).astype(np.float32)
        else:
            dense_vectors[word_key] = np.zeros((0, hidden_size), dtype=np.float32)

    contexts = phase1_payload["contexts"]
    for category, named_sets in contexts.items():
        for set_name, words in named_sets.items():
            for word in words.keys():
                key = (category, set_name, word)
                if key not in dense_vectors:
                    dense_vectors[key] = np.zeros((0, hidden_size), dtype=np.float32)
                    _ = stats[key]

    return dense_vectors, stats, hidden_size


def make_npz_key(category: str, set_name: str, word: str) -> str:
    return f"{category}__{set_name}__{word}"


def build_manifest(
    phase1_payload: Dict[str, object],
    model_name: str,
    hidden_size: int,
    embedding_file: Path,
    vectors: Dict[Tuple[str, str, str], Any],
    stats: Dict[Tuple[str, str, str], Dict[str, int]],
) -> Dict[str, object]:
    words_manifest: Dict[str, Dict[str, Dict[str, Dict[str, object]]]] = {}
    for category, named_sets in phase1_payload["contexts"].items():
        words_manifest.setdefault(category, {})
        for set_name, words in named_sets.items():
            words_manifest[category].setdefault(set_name, {})
            for word in words.keys():
                key_tuple = (category, set_name, word)
                arr = vectors[key_tuple]
                words_manifest[category][set_name][word] = {
                    "npz_key": make_npz_key(*key_tuple),
                    "vector_count": int(arr.shape[0]),
                    "vector_dim": int(arr.shape[1]),
                    "sentences_seen": int(stats[key_tuple]["sentences_seen"]),
                    "occurrences_mapped": int(stats[key_tuple]["occurrences_mapped"]),
                    "occurrences_unmapped": int(stats[key_tuple]["occurrences_unmapped"]),
                }

    return {
        "phase": 2,
        "description": "Contextual embedding extraction using average of last four BERT layers",
        "model_name": model_name,
        "hidden_size": hidden_size,
        "embedding_file": str(embedding_file),
        "words": words_manifest,
    }


def save_outputs(
    output_npz: Path,
    output_manifest: Path,
    manifest: Dict[str, object],
    vectors: Dict[Tuple[str, str, str], Any],
) -> None:
    np, _, _ = ensure_dependencies()
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    npz_arrays = {make_npz_key(*key): value for key, value in vectors.items()}
    np.savez_compressed(output_npz, **npz_arrays)

    with output_manifest.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2: extract contextual BERT embeddings for Phase 1 contexts."
    )
    parser.add_argument(
        "--input-context-file",
        type=Path,
        default=Path("outputs") / "task1_phase1_contexts.json",
        help="Phase 1 context JSON path",
    )
    parser.add_argument(
        "--output-embeddings-file",
        type=Path,
        default=Path("outputs") / "task2_phase2_embeddings.npz",
        help="Output NPZ containing contextual vectors",
    )
    parser.add_argument(
        "--output-manifest-file",
        type=Path,
        default=Path("outputs") / "task2_phase2_manifest.json",
        help="Output JSON metadata manifest for extracted vectors",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="bert-base-uncased",
        help="Hugging Face model name",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device preference",
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Only verify dependency availability and exit",
    )

    args = parser.parse_args()

    try:
        if args.check_env:
            ensure_dependencies()
            print("Phase 2 dependencies are available.")
            return

        phase1_payload = load_phase1_contexts(args.input_context_file)
        vectors, stats, hidden_size = extract_contextual_vectors(
            phase1_payload=phase1_payload,
            model_name=args.model_name,
            device_preference=args.device,
        )
        manifest = build_manifest(
            phase1_payload=phase1_payload,
            model_name=args.model_name,
            hidden_size=hidden_size,
            embedding_file=args.output_embeddings_file,
            vectors=vectors,
            stats=stats,
        )
        save_outputs(
            output_npz=args.output_embeddings_file,
            output_manifest=args.output_manifest_file,
            manifest=manifest,
            vectors=vectors,
        )
    except RuntimeError as exc:
        raise SystemExit(f"Dependency error: {exc}") from exc

    print(f"Wrote Phase 2 embeddings to {args.output_embeddings_file}")
    print(f"Wrote Phase 2 manifest to {args.output_manifest_file}")


if __name__ == "__main__":
    main()

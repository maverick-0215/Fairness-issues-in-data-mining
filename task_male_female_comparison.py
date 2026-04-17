import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import requests
import torch
from transformers import AutoModel, AutoTokenizer

from task1_phase1 import DEFAULT_CONCEPT_SETS, build_word_index, normalize_text, split_sentences
from task2_phase2 import make_npz_key
from task4_phase4 import compute_bias_rows


FEMALE_AUTHOR_IDS = [1342, 161, 158, 121, 141, 105, 1260, 768, 519, 84, 145, 6688, 550, 514, 284]
MALE_AUTHOR_IDS = [98, 766, 1400, 46, 730, 1661, 2852, 76, 74, 2701, 219, 345, 844, 174, 35]


def gutenberg_url(book_id: int) -> str:
    return f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"


def strip_gutenberg_boilerplate(text: str) -> str:
    start_pat = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)
    end_pat = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)

    start_match = start_pat.search(text)
    end_match = end_pat.search(text)
    if start_match and end_match and start_match.end() < end_match.start():
        text = text[start_match.end() : end_match.start()]
    return text


def clean_book_text(text: str) -> str:
    text = normalize_text(text)
    text = strip_gutenberg_boilerplate(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def download_books(book_ids: Iterable[int], timeout_seconds: int = 30, retries: int = 3) -> Dict[int, str]:
    downloaded: Dict[int, str] = {}
    session = requests.Session()
    headers = {"User-Agent": "fairml_bert_male_female_audit/1.0"}

    for book_id in book_ids:
        url = gutenberg_url(book_id)
        last_error = None
        for attempt in range(1, retries + 1):
            try:
                response = session.get(url, headers=headers, timeout=timeout_seconds)
                response.raise_for_status()
                downloaded[book_id] = clean_book_text(response.text)
                last_error = None
            except Exception as exc:
                last_error = exc
            finally:
                # Required: pause between every request to Gutenberg.
                time.sleep(2)

            if last_error is None:
                break
        if last_error is not None:
            raise RuntimeError(f"Failed to download Gutenberg ID {book_id}: {last_error}") from last_error

    return downloaded


def save_downloaded_books(downloaded: Dict[int, str], target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for book_id, text in downloaded.items():
        out_file = target_dir / f"pg{book_id}.txt"
        out_file.write_text(text, encoding="utf-8")


def extract_contexts_from_corpus(
    corpus_text: str,
    concept_sets: Dict[str, Dict[str, List[str]]],
    max_sentences_per_word: int,
    source_tag: str,
) -> Dict[str, object]:
    concept_words, patterns = build_word_index(concept_sets)
    contexts_by_word: Dict[str, List[Dict[str, str]]] = {cw.word: [] for cw in concept_words}
    seen_sentences: Dict[str, set] = {cw.word: set() for cw in concept_words}
    sentences = split_sentences(corpus_text)

    for sentence in sentences:
        for word, pattern in patterns.items():
            if len(contexts_by_word[word]) >= max_sentences_per_word:
                continue
            if not pattern.search(sentence):
                continue
            dedupe_key = sentence.lower()
            if dedupe_key in seen_sentences[word]:
                continue
            seen_sentences[word].add(dedupe_key)
            contexts_by_word[word].append(
                {
                    "word": word,
                    "sentence": sentence,
                    "source_file": source_tag,
                }
            )

    grouped_contexts: Dict[str, Dict[str, Dict[str, List[Dict[str, str]]]]] = {}
    seen = set()
    for cw in concept_words:
        key = (cw.category, cw.set_name, cw.word)
        if key in seen:
            continue
        seen.add(key)
        grouped_contexts.setdefault(cw.category, {}).setdefault(cw.set_name, {})[cw.word] = contexts_by_word.get(
            cw.word, []
        )

    return {
        "phase": "male_female_custom_phase1",
        "description": "Context extraction from combined corpus string",
        "max_sentences_per_word": max_sentences_per_word,
        "concept_sets": concept_sets,
        "contexts": grouped_contexts,
    }


def build_phase3_style_manifest(
    phase1_payload: Dict[str, object],
    vectors: Dict[Tuple[str, str, str], np.ndarray],
    hidden_size: int,
    model_name: str,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    words_manifest: Dict[str, Dict[str, Dict[str, Dict[str, object]]]] = {}
    centroids_by_npz_key: Dict[str, np.ndarray] = {}

    for category, named_sets in phase1_payload["contexts"].items():
        words_manifest.setdefault(category, {})
        for set_name, words in named_sets.items():
            words_manifest[category].setdefault(set_name, {})
            for word in words.keys():
                key_tuple = (category, set_name, word)
                arr = vectors[key_tuple]
                centroid = np.zeros((hidden_size,), dtype=np.float32) if arr.shape[0] == 0 else arr.mean(axis=0)
                npz_key = make_npz_key(*key_tuple)
                centroids_by_npz_key[npz_key] = centroid.astype(np.float32)
                words_manifest[category][set_name][word] = {
                    "npz_key": npz_key,
                    "centroid_dim": int(hidden_size),
                    "source_vector_count": int(arr.shape[0]),
                    "has_context": bool(arr.shape[0] > 0),
                }

    manifest = {
        "phase": "male_female_custom_phase3",
        "description": "Centroids from male/female corpus comparison run",
        "model_name": model_name,
        "hidden_size": hidden_size,
        "words": words_manifest,
    }
    return manifest, centroids_by_npz_key


def compile_word_patterns(phase1_payload: Dict[str, object]) -> Dict[str, re.Pattern[str]]:
    patterns: Dict[str, re.Pattern[str]] = {}
    for category, named_sets in phase1_payload["contexts"].items():
        for set_name, words in named_sets.items():
            for word in words.keys():
                if word not in patterns:
                    patterns[word] = re.compile(rf"(?<!\w){re.escape(word)}(?!\w)", re.IGNORECASE)
    return patterns


def flatten_sentence_assignments(phase1_payload: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    sentence_map: Dict[str, Dict[str, object]] = {}
    for category, named_sets in phase1_payload["contexts"].items():
        for set_name, words in named_sets.items():
            for word, items in words.items():
                for item in items:
                    sentence = item["sentence"]
                    sentence_id = f"{item['source_file']}::{sentence}"
                    if sentence_id not in sentence_map:
                        sentence_map[sentence_id] = {"sentence": sentence, "assignments": []}
                    sentence_map[sentence_id]["assignments"].append((category, set_name, word))
    return sentence_map


def overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return start_a < end_b and start_b < end_a


def extract_contextual_vectors_batched(
    phase1_payload: Dict[str, object],
    tokenizer: object,
    model: object,
    device: torch.device,
    batch_size: int = 16,
) -> Tuple[Dict[Tuple[str, str, str], np.ndarray], int]:
    sentence_map = flatten_sentence_assignments(phase1_payload)
    patterns = compile_word_patterns(phase1_payload)
    hidden_size = int(model.config.hidden_size)
    vectors_by_word: Dict[Tuple[str, str, str], List[np.ndarray]] = defaultdict(list)

    sentence_items = list(sentence_map.values())
    with torch.no_grad():
        for i in range(0, len(sentence_items), batch_size):
            chunk = sentence_items[i : i + batch_size]
            chunk_sentences = [x["sentence"] for x in chunk]

            encoded = tokenizer(
                chunk_sentences,
                return_offsets_mapping=True,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )
            offset_mapping = encoded.pop("offset_mapping")
            model_inputs = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**model_inputs)
            hidden_states = outputs.hidden_states
            token_embeddings_batch = torch.stack(hidden_states[-4:], dim=0).mean(dim=0)

            for j, sentence_data in enumerate(chunk):
                sentence = sentence_data["sentence"]
                assignments = sentence_data["assignments"]
                offsets = offset_mapping[j].tolist()
                token_embeddings = token_embeddings_batch[j]

                for category, set_name, word in assignments:
                    spans = [m.span() for m in patterns[word].finditer(sentence)]
                    for span_start, span_end in spans:
                        token_indices = []
                        for idx, (tok_start, tok_end) in enumerate(offsets):
                            if tok_start == tok_end:
                                continue
                            if overlap(span_start, span_end, tok_start, tok_end):
                                token_indices.append(idx)
                        if not token_indices:
                            continue
                        word_vec = token_embeddings[token_indices].mean(dim=0).cpu().numpy().astype(np.float32)
                        vectors_by_word[(category, set_name, word)].append(word_vec)

    dense_vectors: Dict[Tuple[str, str, str], np.ndarray] = {}
    for category, named_sets in phase1_payload["contexts"].items():
        for set_name, words in named_sets.items():
            for word in words.keys():
                key = (category, set_name, word)
                vecs = vectors_by_word.get(key, [])
                dense_vectors[key] = np.stack(vecs).astype(np.float32) if vecs else np.zeros((0, hidden_size), dtype=np.float32)
    return dense_vectors, hidden_size


def run_single_corpus_bias(
    corpus_text: str,
    source_tag: str,
    concept_sets: Dict[str, Dict[str, List[str]]],
    tokenizer: object,
    model: object,
    device_obj: torch.device,
    max_sentences_per_word: int,
) -> pd.DataFrame:
    phase1_payload = extract_contexts_from_corpus(
        corpus_text=corpus_text,
        concept_sets=concept_sets,
        max_sentences_per_word=max_sentences_per_word,
        source_tag=source_tag,
    )
    vectors, hidden_size = extract_contextual_vectors_batched(
        phase1_payload=phase1_payload,
        tokenizer=tokenizer,
        model=model,
        device=device_obj,
    )
    manifest, centroids = build_phase3_style_manifest(
        phase1_payload=phase1_payload,
        vectors=vectors,
        hidden_size=hidden_size,
        model_name=str(getattr(model, "name_or_path", "bert-base-uncased")),
    )
    rows = compute_bias_rows(np=np, manifest=manifest, centroids=centroids)
    return pd.DataFrame(rows)


def compare_female_male_bias(
    female_df: pd.DataFrame,
    male_df: pd.DataFrame,
) -> pd.DataFrame:
    left = female_df[["attribute_word", "attribute_set", "bias_score"]].rename(
        columns={
            "attribute_word": "Attribute_Word",
            "attribute_set": "Category",
            "bias_score": "Female_Author_Bias",
        }
    )
    right = male_df[["attribute_word", "attribute_set", "bias_score"]].rename(
        columns={
            "attribute_word": "Attribute_Word",
            "attribute_set": "Category",
            "bias_score": "Male_Author_Bias",
        }
    )
    merged = left.merge(right, on=["Attribute_Word", "Category"], how="inner")
    merged["Delta"] = merged["Male_Author_Bias"] - merged["Female_Author_Bias"]
    merged = merged.reindex(columns=["Attribute_Word", "Category", "Female_Author_Bias", "Male_Author_Bias", "Delta"])
    merged = merged.sort_values(by="Delta", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Male vs Female author comparative bias wrapper using existing Phase 1-4 functions."
    )
    parser.add_argument("--max-sentences-per-word", type=int, default=300)
    parser.add_argument("--model-name", type=str, default="bert-base-uncased")
    parser.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("outputs") / "male_female_downloaded_books",
        help="Directory to persist downloaded Gutenberg texts",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download and save books without running embedding/bias pipeline",
    )
    parser.add_argument(
        "--book-limit",
        type=int,
        default=0,
        help="Optional limit per group for faster test runs (0 means use all books)",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(args.device)

    female_ids = FEMALE_AUTHOR_IDS[: args.book_limit] if args.book_limit > 0 else FEMALE_AUTHOR_IDS
    male_ids = MALE_AUTHOR_IDS[: args.book_limit] if args.book_limit > 0 else MALE_AUTHOR_IDS

    print(f"Downloading female books: {len(female_ids)} ids", flush=True)
    female_books = download_books(female_ids)
    print(f"Downloading male books: {len(male_ids)} ids", flush=True)
    male_books = download_books(male_ids)

    female_download_dir = args.download_dir / "female"
    male_download_dir = args.download_dir / "male"
    save_downloaded_books(female_books, female_download_dir)
    save_downloaded_books(male_books, male_download_dir)

    female_corpus = "\n\n".join(female_books[bid] for bid in female_ids)
    male_corpus = "\n\n".join(male_books[bid] for bid in male_ids)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "task_female_corpus_combined.txt").write_text(female_corpus, encoding="utf-8")
    (args.output_dir / "task_male_corpus_combined.txt").write_text(male_corpus, encoding="utf-8")

    if args.download_only:
        corpus_meta_json = args.output_dir / "task_male_female_corpus_meta.json"
        with corpus_meta_json.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "female_ids": female_ids,
                    "male_ids": male_ids,
                    "female_books_downloaded": len(female_books),
                    "male_books_downloaded": len(male_books),
                    "female_download_dir": str(female_download_dir),
                    "male_download_dir": str(male_download_dir),
                    "female_char_count": len(female_corpus),
                    "male_char_count": len(male_corpus),
                    "max_sentences_per_word": args.max_sentences_per_word,
                    "model_name": args.model_name,
                },
                f,
                indent=2,
            )
        print(f"Downloaded female books: {len(female_books)}")
        print(f"Downloaded male books: {len(male_books)}")
        print(f"Saved female books to: {female_download_dir}")
        print(f"Saved male books to: {male_download_dir}")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModel.from_pretrained(args.model_name, output_hidden_states=True)
    model.to(device_obj)
    model.eval()

    print("Running female corpus bias pipeline...", flush=True)
    female_bias_df = run_single_corpus_bias(
        corpus_text=female_corpus,
        source_tag="female_corpus",
        concept_sets=DEFAULT_CONCEPT_SETS,
        tokenizer=tokenizer,
        model=model,
        device_obj=device_obj,
        max_sentences_per_word=args.max_sentences_per_word,
    )
    print("Running male corpus bias pipeline...", flush=True)
    male_bias_df = run_single_corpus_bias(
        corpus_text=male_corpus,
        source_tag="male_corpus",
        concept_sets=DEFAULT_CONCEPT_SETS,
        tokenizer=tokenizer,
        model=model,
        device_obj=device_obj,
        max_sentences_per_word=args.max_sentences_per_word,
    )

    comparison_df = compare_female_male_bias(female_df=female_bias_df, male_df=male_bias_df)

    comparison_csv = args.output_dir / "task_male_female_bias_comparison.csv"
    female_csv = args.output_dir / "task_female_author_bias_detail.csv"
    male_csv = args.output_dir / "task_male_author_bias_detail.csv"
    corpus_meta_json = args.output_dir / "task_male_female_corpus_meta.json"

    comparison_df.to_csv(comparison_csv, index=False)
    female_bias_df.to_csv(female_csv, index=False)
    male_bias_df.to_csv(male_csv, index=False)
    with corpus_meta_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "female_ids": female_ids,
                "male_ids": male_ids,
                "female_books_downloaded": len(female_books),
                "male_books_downloaded": len(male_books),
                "female_download_dir": str(female_download_dir),
                "male_download_dir": str(male_download_dir),
                "female_char_count": len(female_corpus),
                "male_char_count": len(male_corpus),
                "max_sentences_per_word": args.max_sentences_per_word,
                "model_name": args.model_name,
            },
            f,
            indent=2,
        )

    print(f"Wrote comparison results: {comparison_csv}")
    print(comparison_df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()

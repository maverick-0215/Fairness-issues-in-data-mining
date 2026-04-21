import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Pattern, Tuple


DEFAULT_CONCEPT_SETS = {
    "target": {
        "male": [
            "man",
            "men",
            "male",
            "he",
            "him",
            "his",
            "boy",
            "father",
            "son",
            "brother",
            "husband",
            "king",
        ],
        "female": [
            "woman",
            "women",
            "female",
            "she",
            "her",
            "hers",
            "girl",
            "mother",
            "daughter",
            "sister",
            "wife",
            "queen",
        ],
    },
    "attribute": {
        "science": [
            "science",
            "mathematics",
            "math",
            "physics",
            "chemistry",
            "experiment",
            "theory",
            "logic",
            "analysis",
            "reason",
            "scholar",
            "laboratory",
        ],
        "arts": [
            "art",
            "music",
            "poetry",
            "dance",
            "drama",
            "painting",
            "literature",
            "beauty",
            "emotion",
            "imagination",
            "song",
            "story",
        ],
        "professions": [
            "engineer",
            "doctor",
            "scientist",
            "professor",
            "lawyer",
            "nurse",
            "teacher",
            "artist",
            "poet",
            "dancer",
            "homemaker",
            "manager",
            "programmer",
            "politician",
            "clerk",
        ],
        "stereotype_traits": [
            "logical",
            "rational",
            "analytical",
            "assertive",
            "ambitious",
            "dominant",
            "strong",
            "emotional",
            "gentle",
            "caring",
            "nurturing",
            "submissive",
            "supportive",
            "sensitive",
        ],
    },
}


@dataclass(frozen=True)
class ConceptWord:
    category: str
    set_name: str
    word: str


def load_concept_sets(config_path: Optional[Path]) -> Dict[str, Dict[str, List[str]]]:
    if not config_path:
        return DEFAULT_CONCEPT_SETS

    with config_path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    return loaded


def build_word_index(
    concept_sets: Dict[str, Dict[str, List[str]]],
) -> Tuple[List[ConceptWord], Dict[str, Pattern[str]]]:
    concept_words: List[ConceptWord] = []
    patterns: Dict[str, Pattern[str]] = {}

    for category, named_sets in concept_sets.items():
        for set_name, words in named_sets.items():
            for word in words:
                normalized = word.strip().lower()
                if not normalized:
                    continue
                concept_words.append(
                    ConceptWord(category=category, set_name=set_name, word=normalized)
                )
                if normalized not in patterns:
                    patterns[normalized] = re.compile(
                        rf"(?<!\w){re.escape(normalized)}(?!\w)", re.IGNORECASE
                    )

    return concept_words, patterns


def iter_corpus_files(corpus_dir: Path) -> Iterable[Path]:
    for txt_file in sorted(corpus_dir.glob("*.txt")):
        if txt_file.name.startswith("_"):
            continue
        yield txt_file


def normalize_text(raw_text: str) -> str:
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def is_filled(contexts: Dict[str, List[Dict[str, str]]], max_sentences: int) -> bool:
    return all(len(items) >= max_sentences for items in contexts.values())


def extract_contexts(
    corpus_dir: Path,
    concept_words: List[ConceptWord],
    patterns: Dict[str, Pattern[str]],
    max_sentences_per_word: int,
) -> Tuple[Dict[str, List[Dict[str, str]]], int]:
    contexts: Dict[str, List[Dict[str, str]]] = {cw.word: [] for cw in concept_words}
    seen_sentences: Dict[str, set[str]] = {cw.word: set() for cw in concept_words}
    files_processed = 0

    for file_path in iter_corpus_files(corpus_dir):
        files_processed += 1
        raw_text = file_path.read_text(encoding="utf-8", errors="ignore")
        cleaned_text = normalize_text(raw_text)
        sentences = split_sentences(cleaned_text)

        for sentence in sentences:
            if is_filled(contexts, max_sentences_per_word):
                return contexts, files_processed

            for word, pattern in patterns.items():
                if len(contexts[word]) >= max_sentences_per_word:
                    continue
                if not pattern.search(sentence):
                    continue

                dedupe_key = sentence.lower()
                if dedupe_key in seen_sentences[word]:
                    continue

                seen_sentences[word].add(dedupe_key)
                contexts[word].append(
                    {
                        "word": word,
                        "sentence": sentence,
                        "source_file": file_path.name,
                    }
                )

    return contexts, files_processed


def regroup_contexts(
    concept_words: List[ConceptWord], contexts_by_word: Dict[str, List[Dict[str, str]]]
) -> Dict[str, Dict[str, Dict[str, List[Dict[str, str]]]]]:
    grouped: Dict[str, Dict[str, Dict[str, List[Dict[str, str]]]]] = {}
    seen = set()

    for cw in concept_words:
        key = (cw.category, cw.set_name, cw.word)
        if key in seen:
            continue
        seen.add(key)
        grouped.setdefault(cw.category, {}).setdefault(cw.set_name, {})[cw.word] = (
            contexts_by_word.get(cw.word, [])
        )
    return grouped


def build_output_payload(
    concept_sets: Dict[str, Dict[str, List[str]]],
    grouped_contexts: Dict[str, Dict[str, Dict[str, List[Dict[str, str]]]]],
    files_processed: int,
    max_sentences_per_word: int,
) -> Dict[str, object]:
    return {
        "phase": 1,
        "description": "Corpus processing for predefined target and attribute concept sets",
        "corpus_files_processed": files_processed,
        "max_sentences_per_word": max_sentences_per_word,
        "concept_sets": concept_sets,
        "contexts": grouped_contexts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task 1 / Phase 1: extract contextual sentences for concept words."
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("gutenberg_indian_literature"),
        help="Path to corpus folder containing .txt books",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("outputs") / "task1_phase1_contexts.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--max-sentences-per-word",
        type=int,
        default=50,
        help="Maximum number of contextual sentences to keep per concept word",
    )
    parser.add_argument(
        "--concept-config",
        type=Path,
        default=None,
        help="Optional JSON config for concept sets",
    )

    args = parser.parse_args()

    concept_sets = load_concept_sets(args.concept_config)
    concept_words, patterns = build_word_index(concept_sets)
    contexts_by_word, files_processed = extract_contexts(
        corpus_dir=args.corpus_dir,
        concept_words=concept_words,
        patterns=patterns,
        max_sentences_per_word=args.max_sentences_per_word,
    )
    grouped_contexts = regroup_contexts(concept_words, contexts_by_word)
    payload = build_output_payload(
        concept_sets=concept_sets,
        grouped_contexts=grouped_contexts,
        files_processed=files_processed,
        max_sentences_per_word=args.max_sentences_per_word,
    )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Wrote Task 1 output to {args.output_file}")
    print(f"Files processed: {files_processed}")


if __name__ == "__main__":
    main()

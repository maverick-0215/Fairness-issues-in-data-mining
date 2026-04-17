# Project Context: Evaluating Sociolinguistic Bias in Domain-Specific BERT Embeddings

## Objective
The goal of this project is to build an algorithmic auditing pipeline that quantifies fairness and sociolinguistic bias in domain-specific language models. Specifically, we are analyzing BERT embeddings trained or fine-tuned on specialized literature (e.g. Indian literature from Project Gutenberg) to see how domain-specific texts alter word associations compared to a mathematically neutral baseline (Distance from Zero).

## Methodology & Architecture
We are measuring "Absolute Bias" by calculating the distance of contextualized word embeddings from a neutral zero-baseline using a variant of the Mean Average Cosine (MAC) metric. 

The pipeline consists of 4 main phases:
1.  **Corpus Processing:** Extracting a maximum of `N` contextual sentences from the target corpus for predefined Target Concept sets (e.g., Male vs. Female attribute words) and Attribute Concept sets (e.g., Science vs. Arts words).
2.  **Contextual Embedding Extraction:** Passing the extracted sentences through a Hugging Face BERT model. We must handle WordPiece subword tokenization to reconstruct whole words, and extract the hidden states from the **average of the last four layers** to capture deep semantic meaning.
3.  **Mean Pooling & Centroid Calculation:** Averaging all the contextual vectors for a specific word across the corpus to create a single, stable "centroid vector" representing its global usage in the text.
4.  **Fairness Quantification:** Calculating the cosine similarity between the centroid vectors of the Target sets and Attribute sets. The bias score is defined as the difference in mean cosine similarity: `Bias(a) = mean(cos(a, X)) - mean(cos(a, Y))`. A score of 0 represents perfect mathematical neutrality.

## Tech Stack Requirements
- Python
- Hugging Face `transformers` (for BERT model and tokenizer)
- PyTorch or TensorFlow (for tensor operations)
- NumPy / SciPy (for cosine similarity and vector math)

## Current Task
Please use this context to help me write modular, functional code for this pipeline, starting with Phase 1.
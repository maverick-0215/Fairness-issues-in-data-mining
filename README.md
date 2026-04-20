# FairML BERT Bias Testing

## Why fairness in word embeddings matters

Word embeddings influence how language models rank, retrieve, and generate text. If embeddings encode social stereotypes, applications built on top of them can amplify those patterns in hiring tools, recommendation systems, search/ranking, summarization, and conversational assistants. Measuring bias is therefore a practical reliability and safety step: it helps identify where model associations are skewed, which terms are most affected, and what needs mitigation before deployment.

This project measures sociolinguistic bias in BERT embeddings using Indian literature from Project Gutenberg.

## Target sets (what they mean)

Target sets are anchor concepts used as comparison groups:

- **Male target set** (words representing male-gendered concepts):  
  `man, men, male, he, him, his, boy, father, son, brother, husband, king`
- **Female target set** (words representing female-gendered concepts):  
  `woman, women, female, she, her, hers, girl, mother, daughter, sister, wife, queen`

## Attribute words used to compute bias (current default run)

Bias is computed for attribute words in four sets:

- **Science**:  
  `science, mathematics, math, physics, chemistry, experiment, theory, logic, analysis, reason, scholar, laboratory`
- **Arts**:  
  `art, music, poetry, dance, drama, painting, literature, beauty, emotion, imagination, song, story`
- **Professions**:  
  `engineer, doctor, scientist, professor, lawyer, nurse, teacher, artist, poet, dancer, homemaker, manager, programmer, politician, clerk`
- **Stereotype traits**:  
  `logical, rational, analytical, assertive, ambitious, dominant, strong, emotional, gentle, caring, nurturing, supportive, sensitive`

## How bias is computed

1. For each word, contextual embeddings are extracted from BERT using the **average of the last 4 layers**.
2. For each word, those contextual vectors are mean-pooled into one **centroid vector**.
3. For each attribute word `a`, compute:

`Bias(a) = mean(cos(a, male target set)) - mean(cos(a, female target set))`

Interpretation:

- **Bias(a) > 0**: stronger association with male target words
- **Bias(a) < 0**: stronger association with female target words
- **|Bias(a)|**: distance from neutrality (0 = mathematically neutral)

## Results

- **Indian literature (full corpus) shows clear directional bias on specific words**:
  - **Female-leaning**: `nurse` (**-0.0745**), `beauty` (**-0.0355**)
  - **Male-leaning**: `politician` (**+0.0587**), `ambitious` (**+0.0555**), `scientist` (**+0.0498**)
  - Profession and stereotype terms show the strongest effects.

- **Male vs Female author comparison** (`task_male_female_bias_comparison.csv`) shows both agreement and disagreement:
  - **Same-direction bias** (both corpora lean similarly):
    - `nurse`: female-leaning in both (Female authors **-0.0705**, Male authors **-0.1014**)
    - `manager`: female-leaning in both (**-0.0386** vs **-0.0055**)
    - `lawyer`: male-leaning in both (**+0.0253** vs **+0.0144**)
  - **Opposite-direction bias** (corpora disagree):
    - `strong`: Female authors **-0.0061**, Male authors **+0.0201**
    - `imagination`: Female authors **-0.0092**, Male authors **+0.0128**
    - `painting`: Female authors **+0.0041**, Male authors **-0.0084**

## Key visualizations

- **Comparative scatter (male-author vs female-author bias per word):** `graph_comparative_scatter.ipynb` -> `outputs\male_female_comparative_scatter.png`
- **Indian literature Figure-7-style bias vs affinity scatter:** `graph_indian_results.ipynb` -> `outputs\indian_results_bias_affinity_scatter.png`

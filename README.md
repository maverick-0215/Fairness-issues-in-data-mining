# FairML BERT Bias Testing

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
  `logical, rational, analytical, assertive, ambitious, dominant, strong, emotional, gentle, caring, nurturing, submissive, supportive, sensitive`

## How bias is computed

1. For each word, contextual embeddings are extracted from BERT using the **average of the last 4 layers**.
2. For each word, those contextual vectors are mean-pooled into one **centroid vector**.
3. For each attribute word `a`, compute:

`Bias(a) = mean(cos(a, male target set)) - mean(cos(a, female target set))`

Interpretation:

- **Bias(a) > 0**: stronger association with male target words
- **Bias(a) < 0**: stronger association with female target words
- **|Bias(a)|**: distance from neutrality (0 = mathematically neutral)

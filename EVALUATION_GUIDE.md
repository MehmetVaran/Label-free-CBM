# UMLS Concept Set Evaluation Guide

## Overview

This guide explains how to evaluate the quality of UMLS-generated concept sets using multiple metrics.

## Evaluation Metrics

### 1. **Overall Score** (0.0 - 1.0)
A weighted combination of all metrics:
- **Coverage** (20%): Number of concepts relative to expected size
- **Quality** (30%): Low noise and duplicate ratios
- **Diversity** (20%): Uniqueness of concepts
- **Relevance** (20%): Semantic similarity to disease
- **Semantic Diversity** (10%): Variety of semantic types

**Interpretation:**
- **0.8-1.0**: Excellent concept set
- **0.6-0.8**: Good concept set
- **0.4-0.6**: Acceptable, may need filtering
- **<0.4**: Poor quality, needs improvement

### 2. **Relevance Score** (Semantic Similarity)
Measures how semantically related concepts are to the disease using sentence embeddings.

**Metrics:**
- Average relevance: Mean similarity across all concepts
- Min/Max relevance: Range of similarities
- Higher scores = more relevant concepts

**Good threshold:** Average relevance > 0.5

### 3. **Quality Metrics**
Assesses concept cleanliness:

- **Noise Count**: Concepts that are metadata, too short/long, or non-meaningful
- **Duplicate Count**: Case-insensitive duplicates
- **Noise Ratio**: Proportion of noise (lower is better, <0.1 is good)
- **Duplicate Ratio**: Proportion of duplicates (lower is better, <0.05 is good)

### 4. **Diversity Score**
Measures concept uniqueness:

- **Diversity Score**: Ratio of unique to total concepts (1.0 = all unique)
- **Word Overlap**: Average word overlap between concepts (lower = more diverse)

**Good threshold:** Diversity score > 0.8

### 5. **Semantic Type Analysis**
Distribution of UMLS semantic types:

- **Unique Types**: Number of different semantic types
- **Type Distribution**: Count of each semantic type
- More diverse types = better coverage

**Good threshold:** 3-10 unique semantic types

### 6. **Relation Type Analysis**
Distribution of UMLS relation types (RO, RB, PAR, etc.):

- Shows what types of relationships are captured
- More diverse relations = better concept expansion

## Usage

### Basic Evaluation

```bash
# Evaluate a single disease
python3 evaluate_umls_concepts.py data/umls_concepts/Pneumonia_20251228_171536.json --disease "Pneumonia"

# Evaluate all diseases in a file
python3 evaluate_umls_concepts.py data/umls_concepts.json

# Save report to file
python3 evaluate_umls_concepts.py data/umls_concepts.json --output evaluation_report.txt

# Faster evaluation (without semantic similarity)
python3 evaluate_umls_concepts.py data/umls_concepts.json --no-embeddings
```

### Interpreting Results

1. **Check Overall Score**: Primary quality indicator
2. **Review Quality Metrics**: Identify noise and duplicates
3. **Examine Relevance**: Ensure concepts are disease-related
4. **Assess Diversity**: Verify good coverage without redundancy
5. **Semantic Types**: Confirm appropriate medical categories

## Recommended Thresholds

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| Overall Score | >0.7 | 0.5-0.7 | <0.5 |
| Average Relevance | >0.5 | 0.3-0.5 | <0.3 |
| Noise Ratio | <0.1 | 0.1-0.2 | >0.2 |
| Duplicate Ratio | <0.05 | 0.05-0.15 | >0.15 |
| Diversity Score | >0.8 | 0.6-0.8 | <0.6 |
| Unique Semantic Types | 3-10 | 1-3 or >10 | 0 or >15 |

## Improving Concept Sets

Based on evaluation results:

1. **Low Relevance**: Filter concepts with low semantic similarity
2. **High Noise**: Remove metadata, non-English, or malformed concepts
3. **High Duplicates**: Deduplicate case-insensitively
4. **Low Diversity**: Remove overly similar concepts
5. **Poor Semantic Coverage**: Adjust relation types or expand search

## Example Output

```
================================================================================
UMLS Concept Set Evaluation Report: Pneumonia
================================================================================

SUMMARY METRICS
--------------------------------------------------------------------------------
Total Concepts: 91
Unique Concept Names: 91
Candidate Concepts: 5
Related Concepts: 234

OVERALL SCORE
--------------------------------------------------------------------------------
Overall Score: 0.742 / 1.0

Component Scores:
  - Coverage: 1.000
  - Quality: 0.850
  - Diversity: 1.000
  - Relevance: 0.623
  - Semantic_diversity: 0.100

QUALITY METRICS
--------------------------------------------------------------------------------
Noise Count: 8
Quality Concepts: 83
Duplicates: 0
Noise Ratio: 8.79%
Duplicate Ratio: 0.00%
```

## Integration with Existing Tools

The evaluation framework can be integrated with `conceptset_utils.py` functions:

- Use `filter_too_similar()` to reduce duplicates
- Use `remove_too_long()` to filter noise
- Use `filter_too_similar_to_cls()` to improve relevance


# Wi-Fi Troubleshooting RAG Research

## Overview

This repository contains an experimental research project investigating whether Retrieval-Augmented Generation (RAG) improves the accuracy and groundedness of LLM-generated Wi-Fi troubleshooting procedures.

The project compares two conditions using the same LLM and the same set of Wi-Fi troubleshooting cases:

- **No-RAG:** The LLM generates a troubleshooting procedure without retrieved documentation.
- **RAG:** Relevant troubleshooting documentation is retrieved and provided to the LLM before generation.

The goal is to evaluate whether retrieval improves the alignment of generated troubleshooting procedures with technical evidence.

---

## Research Question

> Does retrieval augmentation improve the accuracy and groundedness of LLM-generated Wi-Fi troubleshooting procedures?

---

## Experimental Design

The experiment uses the same:

- Wi-Fi troubleshooting test cases
- LLM
- evaluation framework

for both experimental conditions.

```text
                    Wi-Fi Test Cases
                           |
              +------------+------------+
              |                         |
            No-RAG                     RAG
              |                         |
              |                  Document Retrieval
              |                         |
              |                         ↓
              |                   Retrieved Evidence
              |                         |
              +------------+------------+
                           |
                          LLM
                           |
                Troubleshooting Procedure
                           |
                     Evaluation

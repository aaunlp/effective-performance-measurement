# Effective Performance Measurement: Challenges and Opportunities in KPI Extraction from Earnings Calls

[![Hugging Face Data](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/AAU-NLP/effective-performance-measurement/tree/main)
> **Authors:** Rasmus T. Aavang, Rasmus Tjalk-Bøggild, Alexandre Iolov, Giovanni Rizzi, Mike Zhang, Johannes Bjerva  
> **Accepted at:** ACL 2026 (Industry Track)

This repository contains the code, to do the ablations in the paper between different models for SEC filings.
## 📋 Abstract
Earnings calls are a key source of financial information about public companies. However, extracting information from these calls is difficult. Unlike the templatic filings required by the U.S. Securities and Exchange Commission (SEC) to report a company's financial situation, earnings conference calls have no built-in labels, are unstructured, and feature conversational language. 

We explore this challenging domain by assessing the information captured by models trained on SEC filings and in-context learning methods. To establish a baseline, we first evaluate the generalization capabilities of SEC-trained models across established SEC datasets. To support our investigation, we introduce three novel benchmarks: **(1) SEC Filings Benchmark (SECB)**, **(2) Earnings Calls Benchmark (ECB)**, and **(3) ECB-A**, a subset with 2,460 expert annotation groups to support our qualitative analysis. We find that encoder-based models struggle with the domain shift. Finally, we propose a system utilizing LLMs to perform open-ended extraction from unstructured call transcripts, verified by human evaluation (79.7% precision), providing a baseline for this valuable domain through the consistent tracking of emergent KPIs.

---

## 🗄️ Datasets
To support research into financial NLP and KPI extraction, we introduce three new benchmarks and a gold-standard tracking set derived from 20 S&P 500 companies (spanning 2023-2024). 

**Access the datasets here:** 🤗 [AAU-NLP/effective-performance-measurement](https://huggingface.co/datasets/AAU-NLP/effective-performance-measurement/tree/main)

* **`SECB.json` (SEC Filings Benchmark):** 40,661 context-rich chunks from SEC filings with 77,677 regex-labeled entities.
* **`ECB.json` (Earnings Call Benchmark):** 10,477 raw, unannotated conversational text chunks extracted from corporate earnings calls.
* **`ECB-A.json` (Annotated Subset):** An expert-annotated subset of ECB featuring 587 chunks and 2,460 entities used for evaluating LLM extraction.
* **`gold_standard_traceable.jsonl`:** 1,323 verified, post-hoc semantic clusterings of KPIs to track emergent metrics across multiple quarters.

---

## ⚙️ Installation & Setup
We use [`uv`](https://github.com/astral-sh/uv) for fast and reliable Python package management.

```bash
git clone [https://github.com/AAU-NLP/effective-performance-measurement.git](https://github.com/AAU-NLP/effective-performance-measurement.git)
cd effective-performance-measurement

# Install dependencies and set up the environment and run the code with 
uv sync
uv run ner

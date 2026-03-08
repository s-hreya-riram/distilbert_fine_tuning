# ESG Claim Classifier

A lightweight NLP pipeline that classifies corporate sustainability statements as **quantified** (backed by measurable targets or results) or **vague** (aspirational language without concrete numbers). Built to explore weak supervision, DistilBERT fine-tuning, and production model serving.

---

## Motivation

Corporate sustainability reports vary widely in rigour. A company claiming *"we are committed to reducing food waste"* is saying something categorically different from *"we reduced food waste by 32% against our 2019 baseline, diverting 4,200 tonnes from landfill in FY2023."* Distinguishing the two at scale is useful for ESG analysts, sustainability researchers, and disclosure screening tools.

The central engineering question this project explores if a small, fast, locally-served model like DistilBERT can match the classification quality of an LLM API such as OpenAI API (gpt-4o-mini) at a fraction of the latency and cost.

---

## Pipeline Overview

```
Module 1 — Data Pipeline
    climatebert/climate_detection corpus
    → clean, deduplicate
    → sentences_raw.csv

Module 2 — Silver Labeling
    sentences_raw.csv
    → GPT-4o-mini weak supervision
    → sentences_labeled.csv (text, label, reason, confidence)

Module 3 — Fine-Tuning
    sentences_labeled.csv
    → DistilBERT fine-tuned on silver labels
    → model checkpoint

Module 4 — Serving + Benchmarking
    model checkpoint
    → FastAPI endpoint
    → latency benchmark vs GPT-4o-mini
```

---

## Results

### Classification Performance (Eval Set, 340 examples)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Vague | 0.97 | 0.93 | 0.95 |
| Quantified | 0.85 | 0.93 | 0.89 |
| **Macro avg** | **0.91** | **0.93** | **0.92** |

- Dataset: 1,700 sentences from a HuggingFace dataset, `climatebert/climate_detection` that we silver labeled via GPT-4o-mini
- Train/eval split: 80/20, stratified by label
- Class distribution: 70% vague, 30% quantified

### Inference Benchmark

| Model | Median Latency | p95 Latency | Cost per 1,000 |
|---|---|---|---|
| DistilBERT (local) | 13.6 ms | 15.9 ms | $0.00 |
| GPT-4o-mini (API) | 1,426.8 ms | 1,897.4 ms | ~$0.15 |

**DistilBERT is 105x faster than GPT-4o-mini at effectively zero marginal inference cost.**

---

## Design Decisions

**Why weak supervision instead of manual labeling?**
GPT-4o-mini generates silver labels at negligible cost (~$0.05 for 1,700 sentences). This is a standard industry technique — large models as labeling oracles, small models as production classifiers. A random sample of 50 labels was manually verified to validate label quality before training.

**Why DistilBERT?**
40% smaller and faster than BERT-base, with comparable performance on classification tasks. Well-suited for fine-tuning on small datasets and cheap to serve locally or on modest cloud hardware.

**Why not just call the LLM API in production?**
The benchmark answers this directly: 105x latency reduction, zero marginal cost per inference, no data leaving your infrastructure. For any application running at meaningful volume, the fine-tuned model dominates on every practical dimension.

**Labeling note**
This is fine-tuning on LLM-generated silver labels, not knowledge distillation in the strict sense. True distillation would use the teacher model's soft probability outputs (logits) rather than hard labels — a natural extension once logprob access is available via the API.

---

## Repo Structure

```
├── data_pipeline.py           # Load, clean, deduplicate corpus
├── silver_labeling.py         # GPT-4o-mini labeling pipeline
├── fine_tuning.py             # DistilBERT fine-tuning
├── serving.py                 # FastAPI inference endpoint
├── benchmarking.py            # Latency comparison
├── constants.py               # Shared config (model path etc.)
├── requirements.txt
└── README.md
```

---

## Quickstart

**1. Install dependencies**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Set up environment**
```bash
cp .envexample .env
# Add your OPENAI_API_KEY to .env
```

**3. Run the pipeline**
```bash
# Build dataset
python data_pipeline.py

# Generate silver labels (requires OpenAI API key)
python silver_labeling.py

# Fine-tune DistilBERT
python fine_tuning.py

# Start serving endpoint
python -m uvicorn serving:app --reload

# Run benchmark (run in a second terminal with the serving endpoint up and running in the first one)
python benchmarking.py
```

**4. Test the endpoint manually**
```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "We reduced food waste by 32% against our 2019 baseline."}'
```

Expected response:
```json
{
  "label": "quantified",
  "confidence": 0.9821,
  "latency_ms": 13.4
}
```

---

## Future Work

- Experimenting with another dataset: Replace `climatebert/climate_detection` with CDP food sector disclosures (`data.cdp.net`), Module 1 would require changes to utilize a new dataset.

— Introduce a baseline comparison: Adding a logistic regression baseline would answer whether DistilBERT's complexity is warranted at all, and at what training data size the advantage materialises.

- Distillation: Use OpenAI's `logprobs` endpoint to extract token-level probabilities as soft labels, then train with KL divergence loss instead of cross-entropy. Closer to classical knowledge distillation than hard-label fine-tuning.

---
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import time
from constants import MODEL_PATH

app = FastAPI(title="ESG Classifier")

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

ID2LABEL = {0: "vague", 1: "quantified"}


class Request(BaseModel):
    text: str


class Response(BaseModel):
    label: str
    confidence: float
    latency_ms: float


@app.post("/classify", response_model=Response)
def classify(req: Request):
    start = time.perf_counter()

    inputs = tokenizer(
        req.text, return_tensors="pt", truncation=True, max_length=256
    )
    with torch.no_grad():
        logits = model(**inputs).logits

    pred = torch.argmax(logits, dim=-1).item()
    prob = torch.softmax(logits, dim=-1)[0][pred].item()
    latency_ms = (time.perf_counter() - start) * 1000

    return Response(
        label=ID2LABEL[pred],
        confidence=round(prob, 4),
        latency_ms=round(latency_ms, 2),
    )


@app.get("/health")
def health():
    return {"status": "ok"}
import requests
import statistics
import json
import time
from dotenv import load_dotenv
from openai import OpenAI

# using a small set of ESG-related sentences for benchmarking
TEST_SENTENCES = [
    "We are committed to reducing food waste across our global operations.",
    "We reduced food waste by 32\\% against our 2019 baseline, diverting 4,200 tonnes from landfill.",
    "Sustainability is at the heart of everything we do.",
    "By 2025, we target a 50\\% reduction in food loss across our Tier 1 suppliers.",
    "We continue to work toward a more sustainable supply chain.",
    "Our facility in Rotterdam achieved zero food waste to landfill in FY2023.",
    "We believe in responsible sourcing and environmental stewardship.",
    "Food waste intensity per tonne of production fell 18\\% year-on-year.",
]

N_RUNS = 20   # runs per sentence for stable latency estimates


def benchmark_distilbert():
    latencies = []
    for sentence in TEST_SENTENCES:
        for _ in range(N_RUNS):
            r = requests.post(
                "http://localhost:8000/classify",
                json={"text": sentence}
            )
            latencies.append(r.json()["latency_ms"])

    print("=== DistilBERT (local) ===")
    print(f"  Median latency : {statistics.median(latencies):.1f} ms")
    print(f"  p95 latency    : {sorted(latencies)[int(0.95 * len(latencies))]:.1f} ms")
    print(f"  Cost per 1000  : $0.00  (local inference)")
    return latencies


def benchmark_gpt4o_mini():
    load_dotenv()
    client = OpenAI()
    latencies = []    
    for sentence in TEST_SENTENCES:
        start = time.perf_counter()
        try:
            client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=50,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[{
                    "role": "user",
                    "content": f'Classify as "quantified" or "vague": "{sentence}". JSON only: {{"label": "..."}}'
                }]
            )
            latencies.append((time.perf_counter() - start) * 1000)
        except Exception as e:
            print(f"Error occurred: {e}")
    print("\n=== GPT-4o-mini (API) ===")
    print(f"  Median latency : {statistics.median(latencies):.1f} ms")
    print(f"  p95 latency    : {sorted(latencies)[int(0.95 * len(latencies))]:.1f} ms")
    print(f"  Cost per 1000  : ~$0.15  (API pricing)")
    return latencies



if __name__ == "__main__":
    db_latencies  = benchmark_distilbert()
    gpt_latencies = benchmark_gpt4o_mini()

    speedup = statistics.median(gpt_latencies) / statistics.median(db_latencies)
    print(f"\nDistilBERT is {speedup:.0f}x faster than GPT-4o-mini at inference time")
    print("at effectively zero marginal cost vs ~$0.15 per 1000 API calls")
import json
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from constants import SAMPLED_DATA_PATH, LABELED_DATA_PATH

SYSTEM_PROMPT = """You are an ESG analyst classifying corporate sustainability 
statements. You are precise, consistent, and base judgments only on 
what is explicitly stated in the text."""

USER_PROMPT = """Classify this corporate sustainability statement as either:
- "quantified": contains specific measurable targets or results 
  (percentages, absolute figures, tonnes, years, baselines, named KPIs)
- "vague": aspirational or policy language without concrete numbers

Statement: "{text}"

Reply with JSON only, no preamble:
{{"label": "quantified" or "vague", "reason": "one sentence", "confidence": 0.0-1.0}}"""


def label_sentence(text: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            load_dotenv()
            client = OpenAI() 
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=150,
                temperature=0, 
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT.format(text=text)}
                ]
            )
            result = json.loads(response.choices[0].message.content)

            # Validate expected fields are present
            assert result["label"] in ("quantified", "vague")
            assert "reason" in result
            return result

        except Exception as e:
            if attempt == retries - 1:
                print(f"Failed after {retries} attempts: {e}")
                return {"label": None, "reason": str(e), "confidence": 0.0}
            time.sleep(2 ** attempt)   # exponential backoff


def label_batch(df: pd.DataFrame, output_path) -> pd.DataFrame:
    labels, reasons, confidences = [], [], []

    # Run a sanity check on first 20 before committing to full batch
    print("Running sanity check on first 20 rows...")
    for i, row in df.head(20).iterrows():
        result = label_sentence(row["text"])
        labels.append(result["label"])
        reasons.append(result["reason"])
        confidences.append(result.get("confidence", None))
        print(f"[{i+1}/20] {result['label']:<12} | {row['text'][:80]}...")

    proceed = input("Proceed with full batch? (y/n): ")
    if proceed.lower() != "y":
        print("Aborted.")
        return None

    # --- Full batch ---
    print(f"\nLabeling remaining {len(df) - 20} rows...")
    for i, row in df.iloc[20:].iterrows():
        result = label_sentence(row["text"])
        labels.append(result["label"])
        reasons.append(result["reason"])
        confidences.append(result.get("confidence", None))

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(df)} rows labeled...")

        time.sleep(0.05)   # gentle rate limiting

    df = df.copy()
    df["label"] = labels
    df["reason"] = reasons
    df["confidence"] = confidences

    # Drop any rows where labeling failed
    failed = df["label"].isna().sum()
    if failed:
        print(f"Warning: {failed} rows failed labeling — dropping them")
    df = df.dropna(subset=["label"]).reset_index(drop=True)

    # Save
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df):,} labeled rows to {output_path}")

    # Label distribution — important to check for class imbalance
    print("\nLabel distribution:")
    print(df["label"].value_counts())
    print(df["label"].value_counts(normalize=True).round(3))

    return df


if __name__ == "__main__":
    df = pd.read_csv(SAMPLED_DATA_PATH)
    labeled = label_batch(df, LABELED_DATA_PATH)
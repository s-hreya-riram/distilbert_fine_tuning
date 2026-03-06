import pandas as pd
from datasets import load_dataset
import hashlib

def load_raw(dataset_name: str = "climatebert/climate_detection") -> pd.DataFrame:
    """
    Load the raw dataset and return it as a pandas DataFrame.
    
    Args:
        dataset_name (str): The name of the dataset to load. Default is "climatebert/climate_detection".
        
    Returns:
        pd.DataFrame: A DataFrame containing the raw dataset.
    """
    # Load the dataset using Hugging Face's datasets library
    dataset = load_dataset(dataset_name)

    frames = []

    for split_name, split in dataset.items():
        df_split = pd.DataFrame(split)
        df_split['source_split'] = split_name  # Add a column to indicate the split
        frames.append(df_split)

    # Concatenate all splits into a single DataFrame
    df = pd.concat(frames, ignore_index=True)

    return df

def clean(original_df: pd.DataFrame):
    """
    A sentence that is noisy or too short will confuse the labeler and
    add nothing to training. Each filter below has a concrete rationale.
    """
    original_count = len(original_df)

    df = original_df.copy()

    # Normalise whitespace (PDFs and HTML often leave artifacts)
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()

    # Drop nulls
    df = df.dropna(subset=["text"])

    print(f"Original: {original_count:,} rows")
    print(f"After removing na: {len(df):,} rows")
    # Too short — not a real sentence, probably a heading or table cell
    # Too long — probably a merged paragraph, tokenizer will truncate anyway
    #            but these tend to be low-signal for classification
    word_counts = df["text"].str.split().str.len()
    df = df[(word_counts >= 8)]
    print(f"After word count filter: {len(df):,} rows")

    # Must contain at least one letter (filters out number-only rows)
    df = df[df["text"].str.contains(r"[a-zA-Z]", regex=True)]
    print(f"After letter filter: {len(df):,} rows")

    # Drop sentences with no plausible ESG signal — optional but keeps
    # the dataset focused. Extend this list if you see junk in practice.
    boilerplate_patterns = [
        r"^(table of contents|figure \d+|source:)\s*$",  # only if that's ALL the text
        r"^\d[\d\.\s]+$",                                 # pure numbering rows — keep this
    ]
    combined = "|".join(boilerplate_patterns)
    cleaned_df = df[~df["text"].str.lower().str.match(combined, na=False)]
    removed_df = original_df[~original_df.index.isin(cleaned_df.index)]

    cleaned_count = len(cleaned_df)
    print(f"Cleaned: {original_count:,} → {cleaned_count:,} rows "
          f"({original_count - cleaned_count:,} dropped)")
    # Print the first 5 rows removed for verification
    print("First 5 rows removed:")
    print(removed_df.head())
    return cleaned_df.reset_index(drop=True), removed_df.reset_index(drop=True)

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Exact deduplication via MD5 hash. Fuzzy dedup (e.g. MinHash) is overkill
    for this dataset size and would complicate the pipeline for little gain.
    Drop-duplicates on raw text would miss whitespace/case variants, so we
    normalise first then hash.
    """
    df["_hash"] = (
        df["text"]
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .apply(lambda t: hashlib.md5(t.encode()).hexdigest())
    )
    before = len(df)
    df = df.drop_duplicates(subset="_hash").drop(columns="_hash")
    print(f"Deduplicated: {before:,} → {len(df):,} rows "
          f"({before - len(df):,} duplicates removed)")
    return df.reset_index(drop=True)


def sample(df: pd.DataFrame, n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Cap at n rows for labeling cost control so as to remain within budget for OpenAI calls.
    TODO increase if more training data is needed later
    For climatebert/climate_detection, there are only 1700 rows, so no sampling needed.
    """
    if len(df) <= n:
        print(f"Dataset has {len(df):,} rows — no sampling needed")
        return df
    sampled = df.sample(n=n, random_state=seed)
    print(f"Sampled {n:,} rows from {len(df):,}")
    return sampled.reset_index(drop=True)

def export(df: pd.DataFrame, path) -> None:
    df[["text", "source_split"]].to_csv(path, index=False)
    print(f"Saved {len(df):,} rows to {path}")
    print("\nSample rows:")
    print(df["text"].head(5).to_string(index=False))


if __name__ == "__main__":
    # Load raw dataset
    # Default is "climatebert/climate_detection" for preliminary testing
    # TODO change the dataset name later
    df = load_raw()

    # Clean the dataset
    cleaned_df, removed_df = clean(df)

    # Deduplicate the cleaned dataset
    deduplicated_df = deduplicate(cleaned_df)

    # Sample the deduplicated dataset
    sampled_df = sample(deduplicated_df)

    # Export the final sampled DataFrame to CSV
    export(sampled_df, "data/climatebert/raw/sentences_raw.csv")
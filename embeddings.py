import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")

INPUT_FILE = os.path.join(DATA_DIR, "generated_text.csv")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.npy")
SCORES_FILE = os.path.join(DATA_DIR, "similarity_scores.csv")
PAIRS_FILE = os.path.join(DATA_DIR, "sentence_pairs.csv")

MODEL_NAME = "all-MiniLM-L6-v2"

def find_text_column(df):
    preferred = ["Generated_Text", "generated_text", "text", "Text", "sentence", "Sentence"]
    for col in preferred:
        if col in df.columns:
            return col
    for col in df.columns:
        if df[col].dtype == "object":
            return col
    raise ValueError("No text column was found in generated_text.csv.")

def load_sentences():
    df = pd.read_csv(INPUT_FILE)
    text_col = find_text_column(df)
    texts = (
        df[text_col]
        .dropna()
        .astype(str)
        .str.strip()
    )
    texts = texts[texts.str.len() > 0].drop_duplicates().tolist()

    # Week 4 asks for at least 20 sentences.
    if len(texts) < 20:
        raise ValueError(
            f"Only {len(texts)} usable sentences were found. "
            "Add at least 20 sentences to dataset/generated_text.csv."
        )

    return texts[:20]

def main():
    sentences = load_sentences()

    print(f"Loaded {len(sentences)} sentences.")
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        sentences,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    print("Embedding shape:", embeddings.shape)
    print("Embedding dimension:", embeddings.shape[1])

    np.save(EMBEDDINGS_FILE, embeddings)

    similarity_matrix = cosine_similarity(embeddings)

    # Save complete pairwise similarity matrix in long format.
    rows = []
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            rows.append({
                "sentence_1_id": i + 1,
                "sentence_2_id": j + 1,
                "sentence_1": sentences[i],
                "sentence_2": sentences[j],
                "cosine_similarity": float(similarity_matrix[i, j])
            })

    scores_df = pd.DataFrame(rows).sort_values(
        "cosine_similarity", ascending=False
    )
    scores_df.to_csv(SCORES_FILE, index=False)

    # Save the first 10 highest-scoring pairs separately.
    top_pairs = scores_df.head(10).copy()
    top_pairs.to_csv(PAIRS_FILE, index=False)

    print("\nTop 5 most similar sentence pairs:")
    print(top_pairs.head(5).to_string(index=False))

    print("\nFiles created:")
    print(" - dataset/embeddings.npy")
    print(" - dataset/similarity_scores.csv")
    print(" - dataset/sentence_pairs.csv")

if __name__ == "__main__":
    main()

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

import pickle

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from scipy.sparse import save_npz
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.constants import CONTENT_ATTRIBUTES
from utils.text_cleaning import minimal_clean_text, preprocess_for_tfidf

# načtení proměnných prostředí
load_dotenv(PROJECT_ROOT / ".env")

# inicializace modelů
client = OpenAI()
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# vytvoření složky pro uložení embeddingů a TF-IDF modelů
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)


# výpočet SBERT embeddingů pro více textů
def compute_sbert_embeddings(texts: list[str]) -> np.ndarray:
    texts_safe = [
        "" if pd.isna(text) else str(text)
        for text in texts
    ]

    return sbert_model.encode(
        texts_safe,
        convert_to_numpy=True,
        show_progress_bar=True
    )


# výpočet OpenAI embeddingu pro jeden text
def compute_openai_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )

    return response.data[0].embedding


# výpočet OpenAI embeddingů pro více textů
def compute_openai_embeddings(texts: list[str]) -> np.ndarray:
    embeddings = []

    for i, text in enumerate(texts):
        if i % 50 == 0:
            print(f"  OpenAI embeddings: {i}/{len(texts)}")

        # chybějící text reprezentujeme nulovým vektorem
        if pd.isna(text) or not str(text).strip():
            embeddings.append(np.zeros(1536))
        else:
            embeddings.append(
                compute_openai_embedding(str(text))
            )

    return np.array(embeddings)


if __name__ == "__main__":
    print("Načítám data...")

    # načtení partnerských sylabů
    partner_df = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "partner_syllabi_processed.csv"
    )

    # zpracování jednotlivých obsahových atributů
    for attr in CONTENT_ATTRIBUTES:
        print(f"\nZpracovávám atribut: {attr}")

        # příprava textů pro embedding modely
        texts_clean = partner_df[attr].apply(minimal_clean_text).tolist()

        # příprava textů pro TF-IDF
        texts_tfidf = partner_df[attr].apply(preprocess_for_tfidf).tolist()

        # trénování TF-IDF vectorizeru na partnerských textech
        print("  Trénuji TF-IDF vectorizer...")

        vectorizer = TfidfVectorizer()
        vectorizer.fit(texts_tfidf)

        # uložení TF-IDF vectorizeru
        with open(EMBEDDINGS_DIR / f"tfidf_{attr}.pkl", "wb") as f:
            pickle.dump(vectorizer, f)

        # výpočet a uložení TF-IDF matice partnerských textů
        partner_matrix = vectorizer.transform(texts_tfidf)

        save_npz(
            EMBEDDINGS_DIR / f"tfidf_matrix_{attr}.npz",
            partner_matrix
        )

        # výpočet a uložení SBERT embeddingů
        print("  Počítám SBERT embeddingy...")

        sbert_emb = compute_sbert_embeddings(texts_clean)

        np.save(
            EMBEDDINGS_DIR / f"sbert_{attr}.npy",
            sbert_emb
        )

        # výpočet a uložení OpenAI embeddingů
        print("  Počítám OpenAI embeddingy...")

        openai_emb = compute_openai_embeddings(texts_clean)

        np.save(
            EMBEDDINGS_DIR / f"openai_{attr}.npy",
            openai_emb
        )

    print("\nHotovo. Embeddingy uloženy do data/embeddings/")
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pickle

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

from utils.constants import CONTENT_ATTRIBUTES, ATTRIBUTE_WEIGHTS
from utils.text_cleaning import (
    minimal_clean_text,
    preprocess_for_tfidf,
    has_text,
)
from utils.scoring import compute_total_score

# cesta ke složce s předpočítanými modely a embeddingy
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"

# -------------------------------------------------------------------
# Načtení předpočítaných dat
# -------------------------------------------------------------------

# načtení uloženého TF-IDF vectorizeru pro daný atribut
def load_tfidf_vectorizer(attr: str):
    with open(EMBEDDINGS_DIR / f"tfidf_{attr}.pkl", "rb") as f:
        return pickle.load(f)


# načtení předpočítané TF-IDF matice partnerských textů
def load_tfidf_matrix(attr: str):
    return load_npz(
        EMBEDDINGS_DIR / f"tfidf_matrix_{attr}.npz"
    )


# načtení předpočítaných embeddingů pro danou metodu a atribut
def load_embeddings(method: str, attr: str) -> np.ndarray:
    return np.load(
        EMBEDDINGS_DIR / f"{method}_{attr}.npy"
    )


# -------------------------------------------------------------------
# Výpočet podobnosti
# -------------------------------------------------------------------

def compute_scores(
    vse_row: pd.Series,
    partner_df: pd.DataFrame,
    method: str,
    compute_query_embedding=None
) -> pd.DataFrame:
    """
    Vypočítá skóre podobnosti mezi zvoleným předmětem
    a všemi partnerskými předměty.

    vse_row: řádek z datasetu VŠE nebo parsovaný RTF sylabus
    partner_df: dataset partnerských předmětů
    method: tfidf, sbert nebo openai
    compute_query_embedding: funkce pro výpočet embeddingu dotazovaného textu (povinná pro metody sbert a openai)
    """

    # základ výsledné tabulky
    results = partner_df[
        [
            "course_code",
            "course_title",
            "university",
            "country",
            "source_url",
        ]
    ].copy()

    # výpočet dílčích skóre pro jednotlivé obsahové atributy
    for attr in CONTENT_ATTRIBUTES:

        vse_text = vse_row.get(attr)

        # pokud dotazovaný předmět atribut neobsahuje,
        # skóre nelze spočítat
        if not has_text(vse_text):
            results[f"{attr}_score"] = np.nan
            continue

        # -----------------------------------------------------------
        # TF-IDF
        # -----------------------------------------------------------

        if method == "tfidf":

            vectorizer = load_tfidf_vectorizer(attr)
            partner_matrix = load_tfidf_matrix(attr)

            vse_processed = preprocess_for_tfidf(vse_text)
            vse_vector = vectorizer.transform([vse_processed])

            scores = cosine_similarity(
                vse_vector,
                partner_matrix
            ).flatten()

        # -----------------------------------------------------------
        # SBERT / OpenAI
        # -----------------------------------------------------------

        else:

            if compute_query_embedding is None:
                raise ValueError(
                    "Pro metodu sbert/openai musí být předána "
                    "funkce compute_query_embedding."
                )

            partner_embeddings = load_embeddings(
                method,
                attr
            )

            vse_emb = compute_query_embedding(
                minimal_clean_text(vse_text)
            )

            if vse_emb is None:
                results[f"{attr}_score"] = np.nan
                continue

            scores = cosine_similarity(
                [vse_emb],
                partner_embeddings
            ).flatten()

        # partneři bez daného atributu dostávají NaN,
        # aby nebyli penalizováni nulovou podobností
        partner_has_text = partner_df[attr].apply(has_text)

        scores = np.where(
            partner_has_text,
            scores,
            np.nan
        )

        results[f"{attr}_score"] = scores

    # ---------------------------------------------------------------
    # Výpočet celkového skóre
    # ---------------------------------------------------------------

    total_scores = []
    available_weights = []
    coverages = []

    total_weight = sum(ATTRIBUTE_WEIGHTS.values())

    for _, row in results.iterrows():

        # sestavení slovníku dílčích skóre
        attr_scores = {
            attr: (
                None
                if pd.isna(row[f"{attr}_score"])
                else row[f"{attr}_score"]
            )
            for attr in CONTENT_ATTRIBUTES
        }

        # výpočet celkového skóre a dostupné váhy
        score, weight = compute_total_score(attr_scores)

        total_scores.append(score)
        available_weights.append(weight)

        # podíl dostupných vah vůči maximální možné váze
        coverages.append(weight / total_weight)

    results["final_score"] = total_scores
    results["available_weight"] = available_weights
    results["coverage"] = coverages

    # seřazení výsledků podle celkového skóre
    return results.sort_values(
        "final_score",
        ascending=False
    )
import pandas as pd
from utils.constants import ATTRIBUTE_WEIGHTS, CONTENT_ATTRIBUTES
from utils.text_cleaning import has_text


def compute_total_score(attribute_scores: dict) -> tuple[float | None, float]:
    """
    Vypočítá celkové skóre podobnosti jako vážený průměr dílčích skóre.
    Váhy se normalizují podle dostupných atributů.

    attribute_scores: slovník {název_atributu: skóre nebo None}
    Vrací: (final_score, available_weight)
    """
    numerator = 0.0
    denominator = 0.0

    for attr in CONTENT_ATTRIBUTES:
        score = attribute_scores.get(attr)
        if score is not None:
            numerator += ATTRIBUTE_WEIGHTS[attr] * score
            denominator += ATTRIBUTE_WEIGHTS[attr]

    if denominator == 0:
        return None, 0.0

    return numerator / denominator, denominator


def get_attribute_coverage(row: pd.Series) -> dict:
    """
    Vrací slovník s informací o dostupnosti každého obsahového atributu.
    """
    return {attr: has_text(row[attr]) for attr in CONTENT_ATTRIBUTES}
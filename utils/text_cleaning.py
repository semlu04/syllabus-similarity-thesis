import re
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# nástroje pro předzpracování textu pro TF-IDF
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# základní technické čištění textu
def clean_base_text(value):
    if pd.isna(value):
        return pd.NA

    text = str(value)

    # sjednocení speciálních mezer a HTML entit
    text = text.replace("\xa0", " ")
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;", "&")

    # sjednocení odrážek
    text = re.sub(r"[•◦▪■]", "-", text)

    # odstranění nadbytečných mezer
    text = re.sub(r"\s+", " ", text).strip()

    # odstranění technicky prázdných hodnot
    if text in {"", "-", ".", "_"}:
        return pd.NA

    return text

# spojení více textových částí do jednoho pole
def join_text_parts(parts):
    cleaned = [clean_base_text(part) for part in parts]
    cleaned = [part for part in cleaned if not pd.isna(part)]

    return " ".join(cleaned) if cleaned else pd.NA

# technická normalizace textu po převodu z RTF
def normalize_rtf_text(text):
    if not text:
        return ""

    # sjednocení speciálních mezer a pomlček
    text = text.replace("\xa0", " ")
    text = text.replace("\u2011", "-")
    text = text.replace("–", "-")
    text = text.replace("—", "-")

    # odstranění znaků vzniklých převodem tabulek z RTF
    text = text.replace("|", "")

    # odstranění přebytečných mezer na začátku a konci řádků
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)

    # sjednocení vícenásobných mezer na jednom řádku
    text = re.sub(r"[ \t]{2,}", " ", text)

    # omezení nadbytečných prázdných řádků
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

# kontrola, zda hodnota obsahuje použitelný text
def has_text(value):
    cleaned = clean_base_text(value)
    return not pd.isna(cleaned)

# minimální čištění textu pro výpočet podobnosti
def minimal_clean_text(text):
    if pd.isna(text):
        return None

    text = str(text)

    # sjednocení vícenásobných mezer
    text = re.sub(r"\s+", " ", text).strip()

    if text == "":
        return None

    return text

# předzpracování textu pro TF-IDF
def preprocess_for_tfidf(text):
    if not has_text(text):
        return ""

    text = str(text).lower()

    # odstranění interpunkce
    text = re.sub(r"[^\w\s]", " ", text)

    # tokenizace textu
    tokens = text.split()

    # odstranění stop slov
    tokens = [token for token in tokens if token not in stop_words]

    # lemmatizace tokenů
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return " ".join(tokens)

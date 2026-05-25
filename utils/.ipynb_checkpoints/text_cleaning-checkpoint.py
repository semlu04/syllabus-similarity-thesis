import re
import pandas as pd

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
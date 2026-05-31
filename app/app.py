import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from app.similarity import compute_scores
from app.rtf_parser import parse_syllabus_from_bytes
from utils.constants import CONTENT_ATTRIBUTES
from utils.text_cleaning import has_text

load_dotenv(PROJECT_ROOT / ".env")

ATTRIBUTE_LABELS = {
    "aims": "zaměření",
    "course_content": "obsah",
    "learning_outcomes": "výsledky učení",
}

ATTRIBUTE_LABELS_FULL = {
    "aims": "Zaměření předmětu",
    "course_content": "Obsah předmětu",
    "learning_outcomes": "Výsledky učení",
}


# -------------------------------------------------------------------
# Inicializace modelů a dat
# -------------------------------------------------------------------

# načtení SBERT modelu
@st.cache_resource
def load_sbert_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# načtení OpenAI klienta
@st.cache_resource
def load_openai_client():
    return OpenAI()


# načtení zpracovaných datasetů
@st.cache_data
def load_data():
    vse_df = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "vse_syllabi_processed.csv"
    )

    partner_df = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "partner_syllabi_processed.csv"
    )

    return vse_df, partner_df


# -------------------------------------------------------------------
# Embedding funkce
# -------------------------------------------------------------------

# výpočet SBERT embeddingu pro dotazovaný text
def get_sbert_embedding(text: str):
    if not text:
        return None

    model = load_sbert_model()

    return model.encode(text)


# výpočet OpenAI embeddingu pro dotazovaný text
def get_openai_embedding(text: str):
    if not text:
        return None

    client = load_openai_client()

    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )

    return response.data[0].embedding


# -------------------------------------------------------------------
# Hlavní aplikace
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Porovnání sylabů předmětů",
    layout="wide"
)

st.title("Porovnání sylabů předmětů")

st.caption(
    "Vyhledávání zahraničních předmětů obsahově podobných "
    "povinným předmětům na VŠE."
)

vse_df, partner_df = load_data()

# -------------------------------------------------------------------
# Panel 1 — výběr předmětu VŠE
# -------------------------------------------------------------------

st.subheader("Předmět na VŠE")

course_code_input = st.text_input(
    "Zadejte kód předmětu",
    placeholder="např. 4IT218"
).strip().upper()

vse_row = None

if course_code_input:
    match = vse_df[
        vse_df["course_code"] == course_code_input
    ]

    if not match.empty:
        vse_row = match.iloc[0]

        st.success(
            f"**{vse_row['course_code']} — {vse_row['course_title']}**"
        )

        # zobrazení dostupnosti obsahových atributů
        coverage_info = []

        for attr in CONTENT_ATTRIBUTES:
            value = vse_row.get(attr)

            if has_text(value):
                coverage_info.append(f"✓ {ATTRIBUTE_LABELS[attr]}")
            else:
                coverage_info.append(f"✗ {ATTRIBUTE_LABELS[attr]}")

        st.caption("  ·  ".join(coverage_info))

    else:
        st.warning(
            f"Kód předmětu **{course_code_input}** nebyl nalezen v databázi."
        )

        with st.expander("Nahrát RTF sylabus z InSIS", expanded=True):
            st.markdown(
                "Pokud váš předmět není v databázi, "
                "nahrajte anglický RTF sylabus přímo z InSIS."
            )

            st.markdown(
                "**Jak stáhnout sylabus z InSIS:**\n"
                "1. V katalogu předmětů InSIS otevřete sylabus předmětu\n"
                "2. Přepněte na anglickou verzi sylabu\n"
                "3. Sjeďte na konec stránky, u typu výstupu přepněte "
                "z PDF na **Dokument RTF** a klikněte na **Zobrazit** "
                "— soubor se automaticky stáhne\n"
                "4. Stažený soubor nahrajte níže"
            )

            uploaded_file = st.file_uploader(
                "Vyberte RTF soubor",
                type=["rtf"]
            )

            if uploaded_file is not None:
                try:
                    file_bytes = uploaded_file.read()
                    parsed = parse_syllabus_from_bytes(file_bytes)
                    vse_row = pd.Series(parsed)

                    st.success(
                        f"Sylabus úspěšně načten: "
                        f"**{parsed.get('course_code', '')} — "
                        f"{parsed.get('course_title', '')}**"
                    )

                    # zobrazení dostupnosti obsahových atributů
                    coverage_info = []

                    for attr in CONTENT_ATTRIBUTES:
                        value = vse_row.get(attr)

                        if has_text(value):
                            coverage_info.append(
                                f"✓ {ATTRIBUTE_LABELS[attr]}"
                            )
                        else:
                            coverage_info.append(
                                f"✗ {ATTRIBUTE_LABELS[attr]}"
                            )

                    st.caption("  ·  ".join(coverage_info))

                except Exception as e:
                    st.error(f"Chyba při zpracování souboru: {e}")

# -------------------------------------------------------------------
# Panel 2 — filtry a nastavení
# -------------------------------------------------------------------

st.subheader("Filtry a nastavení")

col1, col2, col3 = st.columns(3)

with col1:
    universities = ["Všechny"] + sorted(
        partner_df["university"].dropna().unique().tolist()
    )

    selected_university = st.selectbox(
        "Univerzita",
        universities
    )

with col2:
    countries = ["Všechny"] + sorted(
        partner_df["country"].dropna().unique().tolist()
    )

    selected_country = st.selectbox(
        "Země",
        countries
    )

with col3:
    min_score = st.selectbox(
        "Min. skóre podobnosti",
        [0.0, 0.30, 0.40, 0.50, 0.60],
        format_func=lambda x: f"{x:.2f}"
    )

method = st.radio(
    "Metoda výpočtu podobnosti",
    options=["tfidf", "sbert", "openai"],
    format_func=lambda x: {
        "tfidf": "TF-IDF",
        "sbert": "SBERT",
        "openai": "OpenAI embeddings",
    }[x],
    horizontal=True
)

# -------------------------------------------------------------------
# Vyhledávání
# -------------------------------------------------------------------

search_button = st.button(
    "Vyhledat podobné předměty",
    disabled=(vse_row is None)
)

if search_button and vse_row is not None:

    # výběr funkce pro výpočet embeddingu dotazovaného textu
    if method == "sbert":
        embedding_fn = get_sbert_embedding
    elif method == "openai":
        embedding_fn = get_openai_embedding
    else:
        embedding_fn = None

    with st.spinner("Počítám podobnost..."):
        try:
            results = compute_scores(
                vse_row=vse_row,
                partner_df=partner_df,
                method=method,
                compute_query_embedding=embedding_fn
            )

        except Exception as e:
            st.error(f"Chyba při výpočtu: {e}")
            st.stop()

    # aplikace filtru podle univerzity
    if selected_university != "Všechny":
        results = results[
            results["university"] == selected_university
        ]

    # aplikace filtru podle země
    if selected_country != "Všechny":
        results = results[
            results["country"] == selected_country
        ]

    # aplikace filtru podle minimálního skóre
    results = results[
        results["final_score"] >= min_score
    ]

    st.markdown(
        f"Nalezeno **{len(results)}** předmětů "
        f"· seřazeno podle skóre ({method.upper()})"
    )

    # -------------------------------------------------------------------
    # Zobrazení výsledků
    # -------------------------------------------------------------------

    if results.empty:
        st.info("Žádné předměty nevyhověly zadaným kritériím.")

    else:
        for _, row in results.iterrows():

            with st.container(border=True):
                col_meta, col_score = st.columns([4, 1])

                with col_meta:
                    st.markdown(
                        f"**{row['course_title']}**  \n"
                        f"{row['university']} · {row['country']} · "
                        f"`{row['course_code']}`"
                    )

                with col_score:
                    st.metric(
                        "Skóre",
                        f"{row['final_score']:.2f}"
                    )

                # zobrazení dílčích skóre obsahových atributů
                attr_cols = st.columns(3)

                for i, attr in enumerate(CONTENT_ATTRIBUTES):
                    with attr_cols[i]:
                        score = row.get(f"{attr}_score")

                        if pd.isna(score):
                            st.caption(
                                f"{ATTRIBUTE_LABELS_FULL[attr]}  \n"
                                "—  nedostupné"
                            )
                        else:
                            st.caption(
                                f"{ATTRIBUTE_LABELS_FULL[attr]}  \n"
                                f"{score:.2f}"
                            )

                # zobrazení dostupnosti atributů a odkazu na sylabus
                coverage_pct = int(row["coverage"] * 100)
                col_cov, col_link = st.columns([3, 1])

                with col_cov:
                    st.caption(
                        f"Dostupnost atributů: {coverage_pct} %"
                    )

                with col_link:
                    if row.get("source_url"):
                        st.markdown(
                            f"[Zobrazit sylabus]({row['source_url']})"
                        )

    st.caption(
        "⚠️ Výsledné skóre je orientační a nenahrazuje "
        "odborné posouzení garanta předmětu."
    )
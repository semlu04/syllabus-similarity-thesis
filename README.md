# syllabus-similarity-thesis

Repozitář obsahuje zdrojové kódy, datové sady a předpočítané modely použité 
v diplomové práci zaměřené na porovnávání sylabů předmětů mezi VŠE a 
partnerskými univerzitami.

## Struktura projektu

- `parsers/` – scrapery a parsery pro jednotlivé univerzity
- `notebooks/` – vývojové notebooky pro sběr, čištění dat a vývoj pipeline
- `data/` – průběžné, finální datové sady a předpočítané embeddingy
- `utils/` – sdílené pomocné funkce, schéma atributů a výpočet skóre
- `app/` – Streamlit aplikace pro porovnávání sylabů
- `precompute_embeddings.py` – skript pro předpočítání embeddingů a TF-IDF modelů

## Spuštění aplikace

**1. Nainstalujte závislosti**

`pip install -r requirements.txt`

**2. Vytvořte soubor `.env` s OpenAI API klíčem**

`OPENAI_API_KEY=váš_klíč`

**3. Embeddingy**

Předpočítané embeddingy jsou součástí repozitáře ve složce `data/embeddings/`.
Pokud je chcete přegenerovat, spusťte `python precompute_embeddings.py`.

⚠️ Přegenerování OpenAI embeddingů vyžaduje platný API klíč a je zpoplatněno.

**4. Spusťte aplikaci**

`streamlit run app/app.py`

## Použité technologie

- Python
- pandas, numpy, scikit-learn, scipy
- sentence-transformers (SBERT)
- OpenAI API
- Streamlit
- requests, BeautifulSoup, Selenium
- striprtf, nltk

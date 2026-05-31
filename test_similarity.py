import sys
sys.path.append(".")

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from app.similarity import compute_scores

load_dotenv(".env")

vse_df = pd.read_csv("data/processed/vse_syllabi_processed.csv")
partner_df = pd.read_csv("data/processed/partner_syllabi_processed.csv")

vse_row = vse_df[vse_df["course_code"] == "4IT537"].iloc[0]

client = OpenAI()


def get_openai_embedding(text):
    if not text:
        return None

    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )

    return response.data[0].embedding


results = compute_scores(
    vse_row=vse_row,
    partner_df=partner_df,
    method="openai",
    compute_query_embedding=get_openai_embedding
)

print(
    results[
        ["course_code", "course_title", "university", "final_score"]
    ].head(10)
)
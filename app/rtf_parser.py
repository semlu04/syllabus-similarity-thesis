import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import re

import pandas as pd
from striprtf.striprtf import rtf_to_text

from utils.text_cleaning import clean_base_text, normalize_rtf_text


# názvy sekcí extrahovaných ze sylabu VŠE
FIELDS = {
    "course_code": "Course code:",
    "course_title": "Course title in English:",
    "credits": "Number of ECTS credits allocated:",
    "semester": "Semester:",
    "study_level": "Level of course and year of study:",
    "aims": "Aims of the course:",
    "learning_outcomes": "Learning outcomes and competences:",
    "course_content": "Course contents:",
    "assessment_methods": "Assessment methods and criteria:",
    "literature": "Reading:",
}


# -------------------------------------------------------------------
# Pomocné funkce pro extrakci sekcí
# -------------------------------------------------------------------

def extract_single_line(text: str, label: str) -> str:
    """Vrátí první neprázdný řádek za zadaným štítkem."""
    start = text.find(label)

    if start == -1:
        return ""

    rest = text[start + len(label):].splitlines()

    for line in rest:
        line = normalize_rtf_text(line)

        if line:
            return line

    return ""


def extract_between(text: str, start_label: str, all_labels: list[str]) -> str:
    """Vrátí text mezi zadaným štítkem a následujícím známým štítkem."""
    start = text.find(start_label)

    if start == -1:
        return ""

    start = start + len(start_label)

    end_positions = [
        text.find(label, start)
        for label in all_labels
        if label != start_label and text.find(label, start) != -1
    ]

    end = min(end_positions) if end_positions else len(text)

    return normalize_rtf_text(text[start:end])


def remove_after_markers(text: str, markers: list[str]) -> str:
    """Odstraní část textu za prvním nalezeným ukončovacím markerem."""
    if not text:
        return ""

    positions = [
        text.find(marker)
        for marker in markers
        if text.find(marker) != -1
    ]

    if positions:
        text = text[:min(positions)]

    return normalize_rtf_text(text)


def extract_first_number(value: str) -> str:
    """Vrátí první číslo nalezené v textu."""
    match = re.search(r"\d+", str(value))

    return match.group(0) if match else ""


def normalize_semester(value: str) -> str:
    """Převede označení semestru na winter/summer."""
    value = str(value).lower()

    if "ws" in value:
        return "winter"

    if "ss" in value:
        return "summer"

    return ""


def clean_assessment(text: str) -> str:
    """Ponechá pouze položky hodnocení s procentuálním zastoupením."""
    if not text:
        return ""

    cut_markers = [
        "Assessment:",
        "Graded courses",
        "Ungraded courses",
        "Special requirements and details:",
    ]

    text = remove_after_markers(text, cut_markers)
    lines = text.splitlines()
    cleaned = []

    for line in lines:
        line = normalize_rtf_text(line)
        match = re.match(r"(.+?)(\d+\s*%)$", line)

        if match:
            name = match.group(1).strip()
            percent = match.group(2).replace(" ", "")

            if name.lower() != "total":
                cleaned.append(f"{name}: {percent}")

    return "\n".join(cleaned)


# -------------------------------------------------------------------
# Hlavní parsovací funkce
# -------------------------------------------------------------------

def parse_syllabus_from_bytes(file_bytes: bytes) -> dict:
    """
    Parsuje RTF sylabus VŠE z bytes, například ze Streamlit file_uploader.
    Vrací slovník se stejnou strukturou jako vse_syllabi_processed.csv.
    """

    # převod RTF souboru na text
    rtf_content = file_bytes.decode("utf-8", errors="ignore")
    raw_text = normalize_rtf_text(rtf_to_text(rtf_content))

    labels = list(FIELDS.values())

    # výchozí struktura odpovídající zpracovanému datasetu
    row = {
        "university": "Prague University of Economics and Business",
        "country": "Czech Republic",
        "source_url": "",
        "academic_year": "",
        "semester": "",
        "semester_normalized": "",
        "study_level": "",
        "credits": "",
        "credit_system": "ECTS",
        "ects_credits": "",
        "course_title": "",
        "course_code": "",
        "aims": "",
        "course_content": "",
        "learning_outcomes": "",
        "assessment_methods": "",
        "literature": "",
    }

    # pole uložená na jednom řádku za příslušným štítkem
    short_fields = [
        "course_code",
        "course_title",
        "credits",
        "semester",
        "study_level",
    ]

    # textová pole tvořená delšími sekcemi sylabu
    long_fields = [
        "aims",
        "learning_outcomes",
        "course_content",
        "assessment_methods",
        "literature",
    ]

    # extrakce krátkých polí
    for field in short_fields:
        row[field] = extract_single_line(raw_text, FIELDS[field])

    # extrakce delších textových sekcí
    for field in long_fields:
        row[field] = extract_between(raw_text, FIELDS[field], labels)

    # odstranění části o pracovní zátěži, která navazuje na obsah předmětu
    row["course_content"] = remove_after_markers(
        row["course_content"],
        ["Learning activities, teaching methods and workload"],
    )

    # extrakce položek hodnocení s procentuálním zastoupením
    row["assessment_methods"] = clean_assessment(
        row["assessment_methods"]
    )

    # sjednocení kreditů
    row["credits"] = extract_first_number(row["credits"])
    row["ects_credits"] = row["credits"]

    # normalizace úrovně studia
    study_level = row["study_level"].lower()

    if "bachelor" in study_level:
        row["study_level"] = "bachelor"
    elif "master" in study_level:
        row["study_level"] = "master"
    else:
        row["study_level"] = ""

    # normalizace semestru
    row["semester_normalized"] = normalize_semester(row["semester"])

    # technické čištění textových atributů
    for field in [
        "aims",
        "course_content",
        "learning_outcomes",
        "assessment_methods",
        "literature",
    ]:
        cleaned = clean_base_text(row[field])

        if pd.isna(cleaned):
            row[field] = ""
        else:
            row[field] = cleaned

    return row
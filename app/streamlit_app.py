"""Streamlit interface for the tuned champion pipeline.

Launch from the project folder:
    streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from models import TorchTabularClassifier  # noqa: F401  (needed to unpickle the PyTorch pipeline)

bundle = joblib.load(ROOT / "outputs" / "champion.joblib")
defaults = json.loads((ROOT / "outputs" / "app_defaults.json").read_text())
COLS = bundle["columns"]
pipe = bundle["pipeline"]

st.set_page_config(page_title="Gauteng ward need classifier", layout="centered")
st.title("Gauteng ward need classifier")
st.warning(
    "Responsible use: this model is a teaching prototype trained on 469 Gauteng wards from "
    "2011 census-based data. It ranks wards for discussion. It must not decide who receives "
    "water, tankers or repairs, and it does not replace operational judgement."
)

with st.expander("What the inputs mean and what the output says"):
    st.markdown(
        "- **Percentages** are shares of households or dwellings in a ward (0 to 100).\n"
        "- **Output:** *High need* means the model expects the ward's share of women-headed "
        "households with income under R19.6k to be above the threshold "
        f"({bundle['threshold_pct']:.1f}%).\n"
        "- Inputs you leave at the default are set to the development-set median for Gauteng wards."
    )

tab_form, tab_csv = st.tabs(["Enter key values", "Upload a CSV"])
KEY = {
    "pw_00": "Households with piped water inside dwelling (%)",
    "pw_06": "Households with no piped water (%)",
    "dw_07": "Dwellings: informal shack in backyard (%)",
    "dw_08": "Dwellings: informal shack not in backyard (%)",
    "car_00": "Households with a car (%)",
    "lgt_00": "Households using electricity for lighting (%)",
    "total_households": "Total households in ward",
}


def valid_range(col):
    return defaults["min"][col], defaults["max"][col]


SHARE_PREFIXES = ("pw_", "dw_", "car_", "lgt_", "stv_", "lln_", "psa_", "lan_", "pg_")


def check_frame(frame):
    """Return a list of plain-language problems; an empty list means valid.

    Share columns are proportions from 0 to 1, as in the training file.
    """
    problems = []
    missing = [c for c in COLS if c not in frame.columns]
    if missing:
        problems.append(f"Missing columns: {', '.join(missing[:6])}{' ...' if len(missing) > 6 else ''}. "
                        "Use the same column names as the training file.")
    for col in [c for c in COLS if c in frame.columns]:
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.isna().any():
            problems.append(f"Column '{col}' has values that are not numbers.")
        elif col.startswith(SHARE_PREFIXES) and ((values < 0) | (values > 1)).any():
            problems.append(f"Column '{col}' should be a proportion between 0 and 1 (as in the training file).")
    return problems


def predict(frame):
    proba = pipe.predict_proba(frame[COLS])[:, 1]
    return proba


def explain(row, proba):
    """Plain-language reasons from the features the model relies on most."""
    band = "high need" if proba >= 0.5 else "lower need"
    reasons = []
    if row["pw_00"] < 0.5:
        reasons.append("fewer households have piped water inside the dwelling than in a typical Gauteng ward")
    if row["dw_07"] + row["dw_08"] > 0.10:
        reasons.append("a larger share of dwellings are informal than in a typical ward")
    if row["car_00"] < 0.30:
        reasons.append("fewer households own a car than in a typical ward")
    text = "; ".join(reasons) if reasons else "the values are close to a typical Gauteng ward"
    return f"Classified as **{band}** because {text}. This is a statistical pattern, not a cause."


with tab_form:
    values = {}
    cols_ui = st.columns(2)
    for i, (col, label) in enumerate(KEY.items()):
        lo, hi = valid_range(col)
        is_share = col.startswith(SHARE_PREFIXES)
        scale = 100.0 if is_share else 1.0  # the form takes percentages; the model takes proportions
        with cols_ui[i % 2]:
            entered = st.number_input(label, min_value=float(lo * scale), max_value=float(hi * scale),
                                      value=float(np.clip(defaults["median"][col] * scale, lo * scale, hi * scale)),
                                      help=f"Gauteng range in development data: {lo * scale:.2f} to {hi * scale:.2f}")
        values[col] = entered / scale
    if st.button("Classify this ward", type="primary"):
        row = {c: defaults["median"][c] for c in COLS}
        row.update(values)
        frame = pd.DataFrame([row])
        p = float(predict(frame)[0])
        st.metric("Probability of high need", f"{p:.0%}")
        st.write(explain(row, p))

with tab_csv:
    st.caption("One ward per row, with the same column names and units as the training file: "
               "shares are proportions from 0 to 1, counts are whole numbers.")
    upload = st.file_uploader("Choose a CSV file", type=["csv"])
    if upload is not None:
        try:
            frame = pd.read_csv(upload)
        except Exception:
            st.error("The file could not be read as CSV. Check the delimiter and encoding.")
            frame = None
        if frame is not None:
            problems = check_frame(frame)
            if problems:
                for msg in problems:
                    st.error(msg)
            else:
                frame["probability_high_need"] = predict(frame).round(3)
                frame["prediction"] = np.where(frame["probability_high_need"] >= 0.5, "high need", "lower need")
                st.dataframe(frame[["probability_high_need", "prediction"]].head(200))
                st.download_button("Download predictions", frame.to_csv(index=False), "predictions.csv")

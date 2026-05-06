from pathlib import Path

import pandas as pd
import streamlit as st

from src.ingestion import (
    SOURCE_TYPE_OPTIONS,
    normalize_feedback_dataframe,
    read_txt_feedback,
    save_normalized_data,
)


st.set_page_config(page_title="ResidentOps Copilot", layout="wide")

st.title("AI Resident Experience + Operations Copilot")
st.caption("Step 2: data ingestion and normalization")

with st.expander("Unified schema (normalized output)"):
    st.code(
        "source_type, source_id, property_id, unit_id, resident_id, timestamp, text",
        language="text",
    )

st.subheader("Upload resident signals")
col1, col2 = st.columns(2)

with col1:
    source_type = st.selectbox("Source type", SOURCE_TYPE_OPTIONS, index=0)
    property_id = st.text_input("Property ID", value="aker_demo_property_001")

with col2:
    uploaded_file = st.file_uploader(
        "Upload CSV or TXT",
        type=["csv", "txt"],
        help="PDF ingestion can be added next; this step implements CSV/TXT reliably.",
    )

if uploaded_file is None:
    st.info("Upload a file to start ingestion.")
    st.stop()

try:
    file_suffix = Path(uploaded_file.name).suffix.lower()
    if file_suffix == ".csv":
        raw_df = pd.read_csv(uploaded_file)
    elif file_suffix == ".txt":
        raw_df = read_txt_feedback(uploaded_file)
    else:
        st.error("Unsupported file type. Use CSV or TXT.")
        st.stop()
except Exception as exc:  # noqa: BLE001
    st.error(f"Failed to read file: {exc}")
    st.stop()

st.markdown("### Raw data preview")
st.dataframe(raw_df.head(20), use_container_width=True)

result = normalize_feedback_dataframe(
    df=raw_df,
    default_source_type=source_type,
    default_property_id=property_id,
)

st.markdown("### Normalized data preview")
st.dataframe(result.normalized_df.head(20), use_container_width=True)

if result.warnings:
    st.warning("\n".join(result.warnings))
else:
    st.success("Normalization completed with no warnings.")

output_path = save_normalized_data(
    result.normalized_df,
    output_dir=Path("data/processed"),
    stem=Path(uploaded_file.name).stem,
)
st.success(f"Saved normalized file to `{output_path}`")

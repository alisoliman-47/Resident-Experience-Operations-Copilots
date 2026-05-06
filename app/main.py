from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.classification import enrich_with_classification
from src.ingestion import (
    SOURCE_TYPE_OPTIONS,
    normalize_feedback_dataframe,
    read_txt_feedback,
    save_normalized_data,
)


st.set_page_config(page_title="ResidentOps Copilot", layout="wide")

st.title("AI Resident Experience + Operations Copilot")
st.caption("Step 3: ingestion + classification + urgency scoring")

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

classified = enrich_with_classification(result.normalized_df).enriched_df
classified_output_path = save_normalized_data(
    classified,
    output_dir=Path("data/processed"),
    stem=f"{Path(uploaded_file.name).stem}_classified",
)

st.markdown("### Priority queue (classified)")
priority_cols = ["source_id", "timestamp", "category", "sentiment", "urgency", "text"]
st.dataframe(classified[priority_cols].head(30), use_container_width=True)
st.success(f"Saved classified file to `{classified_output_path}`")

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric("Total records", int(len(classified)))
with col_b:
    st.metric("Urgent issues", int((classified["urgency"] == "urgent").sum()))
with col_c:
    st.metric("High + Urgent", int(classified["urgency"].isin(["high", "urgent"]).sum()))

chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    cat_counts = classified["category"].value_counts().reset_index()
    cat_counts.columns = ["category", "count"]
    st.plotly_chart(
        px.bar(cat_counts, x="category", y="count", title="Issue Categories"),
        use_container_width=True,
    )

with chart_col2:
    urgency_counts = classified["urgency"].value_counts().reset_index()
    urgency_counts.columns = ["urgency", "count"]
    urgency_order = ["low", "medium", "high", "urgent"]
    urgency_counts["urgency"] = pd.Categorical(
        urgency_counts["urgency"], categories=urgency_order, ordered=True
    )
    urgency_counts = urgency_counts.sort_values("urgency")
    st.plotly_chart(
        px.bar(urgency_counts, x="urgency", y="count", title="Urgency Distribution"),
        use_container_width=True,
    )

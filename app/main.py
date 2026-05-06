import streamlit as st


st.set_page_config(page_title="ResidentOps Copilot", layout="wide")

st.title("AI Resident Experience + Operations Copilot")
st.caption("Step 1 scaffold: project initialized and ready for data ingestion.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Current Status")
    st.success("Project scaffold complete")
    st.write(
        "- App shell ready\n"
        "- Dependencies defined\n"
        "- Data folders created\n"
        "- Next step: ingestion pipeline"
    )

with col2:
    st.subheader("Roadmap")
    st.write(
        "1. Data ingestion\n"
        "2. Classification + urgency scoring\n"
        "3. RAG Q&A\n"
        "4. Dashboard analytics\n"
        "5. Weekly summary generation"
    )

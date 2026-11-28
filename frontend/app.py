
import os
import streamlit as st
import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:7860").rstrip("/")

st.set_page_config(page_title="Aurora Stays", page_icon="🏨", layout="wide")

with st.sidebar:
    st.header("⚙️ Parameters")
    k = st.slider("Chunks to retrieve (k)", 1, 20, 5)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    max_tokens = st.slider("Max answer tokens", 64, 2048, 512, 64)

st.title("🏨 Aurora Stays")
st.write("Ask a question about AuroraStay hotel policies.")

response_type = st.radio(
    "Response mode",
    ["Answer only", "Answer + Relevant Chunks"],
    horizontal=True,
)
query = st.text_input(
    "Your question",
    placeholder="e.g., Is breakfast included at the Miami Brickell property?",
)

if st.button("Submit", type="primary"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        endpoint = (
            "/v1/answer"
            if response_type == "Answer only"
            else "/v1/answer_with_relevant_chunks"
        )
        payload = {
            "query": query.strip(),
            "k": k,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            with st.spinner("Searching the policy documents and preparing an answer..."):
                response = requests.post(
                    f"{BACKEND_URL}{endpoint}",
                    json=payload,
                    timeout=180,
                )

            if response.ok:
                result = response.json()
                st.subheader("Answer")
                st.write(result.get("answer", "The API returned no answer."))

                if response_type == "Answer + Relevant Chunks":
                    st.subheader("Retrieved policy passages")
                    for i, chunk in enumerate(result.get("relevant_chunks", []), start=1):
                        with st.expander(f"Retrieved passage {i}"):
                            st.write(chunk)
            else:
                try:
                    detail = response.json()
                except ValueError:
                    detail = response.text
                st.error(f"Backend error ({response.status_code}): {detail}")

        except requests.exceptions.RequestException as exc:
            st.error(
                f"Cannot connect to the backend at {BACKEND_URL}. "
                f"Confirm the backend is running and the Codespaces port is Public. "
                f"Details: {exc}"
            )

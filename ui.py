import streamlit as st
import requests

st.set_page_config(
    page_title="Hallucination Insurance",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Hallucination Insurance")
st.subheader("Verify if an AI claim is faithful to its source")

st.divider()

context = st.text_area(
    "📄 Source Context",
    placeholder="Paste the source document or context here...",
    height=200
)

claim = st.text_input(
    "🔍 Claim to Verify",
    placeholder="Enter the claim you want to verify..."
)

if st.button("Verify Claim", type="primary"):
    if not context or not claim:
        st.error("Please provide both context and claim.")
    else:
        with st.spinner("Verifying..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/verify",
                    json={"claim": claim, "context": context}
                )
                result = response.json()
                
                st.divider()
                
                if result['is_faithful']:
                    st.success("✅ FAITHFUL — This claim is supported by the context")
                else:
                    st.error("❌ HALLUCINATION — This claim is not supported by the context")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Confidence Score", f"{result['confidence_score'] * 100:.0f}%")
                with col2:
                    st.metric("Verdict", "Faithful" if result['is_faithful'] else "Hallucination")
                
                st.subheader("Reasoning")
                st.info(result['reasoning'])
                
            except Exception as e:
                st.error(f"Error connecting to API: {e}")
                st.info("Make sure the FastAPI server is running: uvicorn app.main:app --reload")

st.divider()
st.subheader("📋 Batch Verification")
st.caption("Verify multiple claims at once")

batch_claims = st.text_area(
    "Enter claims (one per line)",
    placeholder="The Eiffel Tower is in Paris\nIt was built in 1889\nIt is 500 meters tall",
    height=150
)

if st.button("Verify All Claims", type="secondary"):
    if not context or not batch_claims:
        st.error("Please provide context and at least one claim.")
    else:
        claims_list = [c.strip() for c in batch_claims.split('\n') if c.strip()]
        with st.spinner(f"Verifying {len(claims_list)} claims..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/verify/batch",
                    json={"claims": claims_list, "context": context}
                )
                results = response.json()['results']
                
                st.divider()
                for r in results:
                    if r['is_faithful']:
                        st.success(f"✅ {r['claim']} — {r['confidence_score']*100:.0f}% confident")
                    else:
                        st.error(f"❌ {r['claim']} — {r['confidence_score']*100:.0f}% confident")
                        
            except Exception as e:
                st.error(f"Error: {e}")
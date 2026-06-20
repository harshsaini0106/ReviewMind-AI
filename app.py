import streamlit as st

from workflow import workflow

st.set_page_config(
page_title="ReviewMind AI",
page_icon="",
layout="wide"
)

st.title(" ReviewMind AI")
st.subheader("AI-Powered Review Analysis & Customer Support Agent")

review = st.text_area(
"Enter Customer Review",
height=150,
placeholder="Write a review here..."
)

if st.button("Analyze Review"):

    if review.strip():

        with st.spinner("Analyzing Review..."):

            result = workflow.invoke(
                {
                    "review": review
                }
            )

        st.success("Analysis Complete")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Sentiment",
                result["sentiment"].upper()
            )

        if "diagnosis" in result:

            diagnosis = result["diagnosis"]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Issue Type", diagnosis["issue_type"])

            with col2:
                st.metric("Tone", diagnosis["tone"])

            with col3:
                st.metric("Urgency", diagnosis["urgency"])

        st.subheader("Generated Response")

        st.write(result["response"])

    else:
        st.warning("Please enter a review.")
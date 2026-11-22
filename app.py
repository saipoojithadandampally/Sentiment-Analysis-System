
import streamlit as st
import pandas as pd
import joblib
import re
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis System",
    page_icon="📊",
    layout="wide"
)

# Load saved models
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
model_name = joblib.load("model_name.pkl")

sentiment_mapping = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Prediction function
def predict_sentiment(text):
    cleaned_text = clean_text(text)
    vector = tfidf.transform([cleaned_text])

    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]

    confidence = max(probabilities) * 100

    return sentiment_mapping[prediction], round(confidence, 2)


# Application heading
st.title("📊 Sentiment Analysis System")

st.write(
    "An NLP-based application that analyzes text "
    "and identifies Positive, Negative and Neutral sentiments."
)

st.caption(f"Selected Machine Learning Model: {model_name}")

st.divider()

# Navigation
option = st.sidebar.radio(
    "Choose Analysis",
    ["Single Review Analysis", "Bulk Review Analysis", "About Project"]
)

# Single review analysis
if option == "Single Review Analysis":

    st.header("Analyze a Single Review")

    user_text = st.text_area(
        "Enter your review or sentence:",
        placeholder="Example: I really love this product!"
    )

    if st.button("Analyze Sentiment", type="primary"):

        if user_text.strip():

            sentiment, confidence = predict_sentiment(user_text)

            st.subheader("Analysis Result")

            st.write("**Input Text:**", user_text)

            st.metric("Predicted Sentiment", sentiment)

            st.metric("Prediction Confidence", f"{confidence}%")

            st.progress(confidence / 100)

        else:
            st.warning("Please enter some text.")

# Bulk review analysis
elif option == "Bulk Review Analysis":

    st.header("Analyze Multiple Reviews")

    uploaded_file = st.file_uploader(
        "Upload a CSV file containing reviews",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            df = pd.read_csv(uploaded_file)

            if "text" not in df.columns:
                st.error(
                    "Your CSV must contain a column named 'text'."
                )

            else:
                if st.button("Analyze All Reviews", type="primary"):

                    df = df.dropna(subset=["text"]).copy()

                    if df.empty:
                        st.warning("No valid reviews found.")

                    else:
                        results = df["text"].apply(predict_sentiment)

                        df["Sentiment"] = results.apply(
                            lambda x: x[0]
                        )

                        df["Confidence"] = results.apply(
                            lambda x: x[1]
                        )

                        st.success("Analysis completed!")

                        st.subheader("Analysis Results")

                        st.dataframe(df, use_container_width=True)

                        st.subheader("Sentiment Distribution")

                        counts = df["Sentiment"].value_counts().reset_index()
                        counts.columns = ["Sentiment", "Count"]

                        fig = px.bar(
                            counts,
                            x="Sentiment",
                            y="Count",
                            title="Review Sentiment Distribution"
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        csv_data = df.to_csv(index=False).encode("utf-8")

                        st.download_button(
                            "Download Analysis Results",
                            data=csv_data,
                            file_name="sentiment_analysis_results.csv",
                            mime="text/csv"
                        )

        except Exception as error:
            st.error(f"Unable to process CSV: {error}")

# About project
elif option == "About Project":

    st.header("About the Project")

    st.write("""
    This project uses Natural Language Processing and
    Machine Learning to classify text into three sentiment
    categories:

    - Positive
    - Negative
    - Neutral

    **Technologies Used:**
    - Python
    - Natural Language Processing
    - TF-IDF
    - Logistic Regression
    - Multinomial Naive Bayes
    - Streamlit
    - Pandas
    - Plotly
    """)

    st.write("Selected model:", model_name)

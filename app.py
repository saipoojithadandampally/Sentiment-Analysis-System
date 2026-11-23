from flask import Flask, request, render_template_string
from transformers import pipeline
from pymongo import MongoClient
from datetime import datetime
import pandas as pd
import os

app = Flask(__name__)

# Load Hugging Face model
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model=model_name,
    tokenizer=model_name
)

# MongoDB connection
MONGO_URI = os.environ.get("MONGO_URI")

client = MongoClient(MONGO_URI)
db = client["sentiment_analyzer"]
collection = db["sentiment_records"]


def analyze_review(review):
    prediction = sentiment_analyzer(
        review,
        truncation=True,
        max_length=512
    )[0]

    return {
        "review": review,
        "sentiment": prediction["label"].lower(),
        "confidence": round(prediction["score"] * 100, 2),
        "timestamp": datetime.utcnow()
    }


HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AI-Based Sentiment Analyzer</title>
    <style>
        body {
            font-family: Arial;
            background: #f1f5f9;
            text-align: center;
            padding: 30px;
        }
        .container {
            background: white;
            padding: 30px;
            max-width: 650px;
            margin: auto;
            border-radius: 12px;
        }
        textarea {
            width: 90%;
            height: 100px;
            padding: 12px;
        }
        button {
            background: #2563eb;
            color: white;
            padding: 12px 25px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
        }
        .section {
            margin: 25px 0;
            padding: 15px;
            background: #f8fafc;
            border-radius: 8px;
        }
        .result {
            text-align: left;
            background: #eef2ff;
            padding: 12px;
            margin: 10px;
            border-radius: 6px;
        }
    </style>
</head>
<body>
<div class="container">

    <h1>AI-Based Sentiment Analyzer</h1>
    <p>Analyze customer reviews using NLP and Transformers</p>

    <div class="section">
        <h2>Single Review Analysis</h2>

        <form method="POST">
            <textarea name="review"
                placeholder="Enter your customer review..."
                required></textarea><br><br>

            <button type="submit" name="action" value="single">
                Analyze Sentiment
            </button>
        </form>
    </div>

    <div class="section">
        <h2>Bulk CSV Analysis</h2>
        <p>Upload a CSV file containing a column named review.</p>

        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="file"
                accept=".csv" required><br><br>

            <button type="submit" name="action" value="bulk">
                Analyze CSV
            </button>
        </form>
    </div>

    {% if results %}
        <h2>Analysis Results</h2>

        {% for item in results %}
            <div class="result">
                <p><b>Review:</b> {{ item.review }}</p>
                <p><b>Sentiment:</b> {{ item.sentiment }}</p>
                <p><b>Confidence:</b> {{ item.confidence }}%</p>
            </div>
        {% endfor %}
    {% endif %}

    {% if error %}
        <p style="color:red;">{{ error }}</p>
    {% endif %}

</div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():

    results = []
    error = None

    if request.method == "POST":

        action = request.form.get("action")

        if action == "single":

            review = request.form.get("review", "").strip()

            if review:
                result = analyze_review(review)
                collection.insert_one(result)
                results.append(result)

        elif action == "bulk":

            uploaded_file = request.files.get("file")

            if not uploaded_file or not uploaded_file.filename.endswith(".csv"):
                error = "Please upload a valid CSV file."

            else:
                try:
                    df = pd.read_csv(uploaded_file)

                    if "review" not in df.columns:
                        error = "CSV must contain a column named review."

                    else:
                        for review in df["review"].dropna():
                            review = str(review).strip()

                            if review:
                                result = analyze_review(review)
                                results.append(result)

                        if results:
                            collection.insert_many(results)

                except Exception as e:
                    error = "Unable to process CSV: " + str(e)

    return render_template_string(
        HTML,
        results=results,
        error=error
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

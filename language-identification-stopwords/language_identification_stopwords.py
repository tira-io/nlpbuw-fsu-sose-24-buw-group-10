from pathlib import Path
import re

from tqdm import tqdm
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

if __name__ == "__main__":

    tira = Client()

    # loading validation data (automatically replaced by test data when run on tira)
    text_validation = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training"
    )
    targets_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training"
    )
    print(text_validation)

    lang_ids = [
        "af",
        "az",
        "bg",
        "cs",
        "da",
        "de",
        "el",
        "en",
        "es",
        "fi",
        "fr",
        "hr",
        "it",
        "ko",
        "nl",
        "no",
        "pl",
        "ru",
        "ur",
        "zh",
    ]

    stopwords = {
        lang_id: set(
            (Path(__file__).parent / "stopwords" / f"stopwords-{lang_id}.txt")
            .read_text()
            .splitlines()
        )
        - set(("(", ")", "*", "|", "+", "?"))  # remove regex special characters
        for lang_id in lang_ids
    }

    # Initialize a logistic regression model with character n-grams as features
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(analyzer='char', ngram_range=(2, 5))),
        ('classifier', LogisticRegression())
    ])

    # classifying the data
    predictions = {}
    for lang_id in tqdm(lang_ids):
        lang_stopwords = stopwords[lang_id]
        # Fit the model for each language
        pipeline.fit(text_validation['text'], (targets_validation['lang'] == lang_id).astype(int))
        # Predict probabilities for each text belonging to this language
        predictions[lang_id] = pipeline.predict_proba(text_validation['text'])[:, 1]

    # Convert predictions to DataFrame
    stopword_fractions = pd.DataFrame(predictions, index=text_validation.index)

    # Find the language with the highest probability for each text
    prediction = stopword_fractions.idxmax(axis=1)

    # converting the prediction to the required format
    prediction.name = "lang"
    prediction = prediction.to_frame()
    prediction["id"] = text_validation["id"]
    prediction = prediction[["id", "lang"]]

    # saving the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    prediction.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )

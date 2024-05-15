from pathlib import Path
import re

from tqdm import tqdm
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

class CustomTextFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [
            [len(text), text.count(" "), len(re.findall(r'\w+', text)), len(re.findall(r'[^\s]', text))]
            for text in X
        ]

if __name__ == "__main__":

    tira = Client()

    # loading validation data (automatically replaced by test data when run on tira)
    text_validation = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training"
    )
    #print(text_validation)
    
    targets_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training"
    )
    #print(targets_validation)

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

    # Defining pipeline for text processing and classification
    text_processing_pipeline = Pipeline([
        ('features', FeatureUnion([
            ('tfidf_word', TfidfVectorizer(analyzer='word', stop_words='english')),
            ('tfidf_char', TfidfVectorizer(analyzer='char')),
            ('text_features', CustomTextFeatures())
        ])),
        ('scaler', StandardScaler(with_mean=False)),  # Scale features
        ('classifier', LogisticRegression())  # Logistic Regression classifier
    ])

    # Fit pipeline on training data
    text_processing_pipeline.fit(text_validation['text'], targets_validation['lang'])

    # Predict on validation data
    predictions = text_processing_pipeline.predict(text_validation['text'])

    # Create DataFrame for prediction
    prediction_df = pd.DataFrame({'id': text_validation['id'], 'lang': predictions})

    # saving the prediction
    output_dir = get_output_directory(str(Path(__file__).parent))
    prediction_df.to_json(
        Path(output_dir) / "predictions.jsonl", orient="records", lines=True
    )

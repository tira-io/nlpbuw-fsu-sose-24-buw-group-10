from pathlib import Path
import pandas as pd
from tqdm import tqdm
from langdetect import detect_langs
from langdetect.lang_detect_exception import LangDetectException  # Importing LangDetectException
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

    # classifying the data using langdetect
    predictions = []
    for index, row in tqdm(text_validation.iterrows(), total=len(text_validation)):
        try:
            lang = detect_langs(row["text"])  # Changed to detect_langs
        except LangDetectException:
            # Handle the case where there are no features in the text
            lang = 'unknown'
        predictions.append({"id": row["id"], "lang": lang})

    # saving the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    pd.DataFrame(predictions).to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )

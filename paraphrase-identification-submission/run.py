from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import pandas as pd

if __name__ == "__main__":

    # Load the data
    tira = Client()
    df = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "paraphrase-identification-validation-20240515-training"
    ).set_index("id")
    print(df)

    # Load pre-trained model
    model = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2')

    # Compute embeddings for the sentences
    embeddings1 = model.encode(df['sentence1'].tolist(), convert_to_tensor=True)
    embeddings2 = model.encode(df['sentence2'].tolist(), convert_to_tensor=True)

    # Compute cosine similarity scores
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2).cpu().numpy()

    # Determine labels based on a threshold
    threshold = 0.8  # This threshold can be adjusted based on validation performance
    df["label"] = (cosine_scores.diagonal() >= threshold).astype(int)

    # Drop unnecessary columns and reset index
    df = df.drop(columns=["sentence1", "sentence2"]).reset_index()

    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )

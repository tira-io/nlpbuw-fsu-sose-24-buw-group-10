from pathlib import Path
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line.strip()) for line in file]

def save_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for entry in data:
            file.write(json.dumps(entry) + '\n')

def predict_ner(sentences, model, tokenizer):
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    predictions = []
    for sentence in sentences:
        ner_results = nlp(sentence)
        tags = ["O"] * len(sentence.split())
        for entity in ner_results:
            start_idx = len(" ".join(sentence.split()[:entity['start']]).split())
            end_idx = len(" ".join(sentence.split()[:entity['end']]).split())
            tags[start_idx] = f"B-{entity['entity_group']}"
            for i in range(start_idx + 1, end_idx):
                tags[i] = f"I-{entity['entity_group']}"
        predictions.append(tags)
    return predictions

if __name__ == "__main__":
    tira = Client()

    # loading validation data (automatically replaced by test data when run on tira)
    text_validation = tira.pd.inputs("nlpbuw-fsu-sose-24", "ner-validation-20240612-training")
    targets_validation = tira.pd.truths("nlpbuw-fsu-sose-24", "ner-validation-20240612-training")

    # Load pre-trained model and tokenizer
    model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    # Prepare sentences
    sentences = text_validation['sentence'].tolist()

    # Predict NER tags
    predicted_tags = predict_ner(sentences, model, tokenizer)

    # Create predictions DataFrame
    predictions = text_validation.copy()
    predictions['tags'] = predicted_tags
    predictions = predictions[['id', 'tags']]

    # Saving the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    predictions.to_json(Path(output_directory) / "predictions.jsonl", orient="records", lines=True)

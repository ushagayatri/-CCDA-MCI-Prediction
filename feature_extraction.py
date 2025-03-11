import spacy
import numpy as np
import pandas as pd

nlp = spacy.load("en_core_web_sm")

def extract_features(text):
    doc = nlp(text)
    return {
        "num_tokens": len(doc),
        "num_nouns": sum(1 for token in doc if token.pos_ == "NOUN"),
        "num_verbs": sum(1 for token in doc if token.pos_ == "VERB"),
        "type_token_ratio": len(set(token.text for token in doc)) / len(doc) if len(doc) > 0 else 0,
        "proposition_density": sum(1 for token in doc if token.dep_ in ["advmod", "prep"]) / len(doc) if len(doc) > 0 else 0
    }

df = pd.read_csv("data/processed/transcriptions.csv")
df_features = pd.DataFrame(df["transcription"].apply(lambda x: extract_features(str(x)))).T
df_features.to_csv("data/processed/linguistic_features.csv", index=False)

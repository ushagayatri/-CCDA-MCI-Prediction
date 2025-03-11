import requests
import json

def get_conceptnet_expansion(word):
    url = f"http://api.conceptnet.io/c/en/{word}"
    response = requests.get(url).json()
    related_concepts = [edge["end"]["label"] for edge in response["edges"] if "end" in edge]
    return list(set(related_concepts))

word = "cat"
concept_expansions = get_conceptnet_expansion(word)
print(f"ConceptNet expansions for '{word}': {concept_expansions}")

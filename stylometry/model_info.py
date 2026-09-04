import spacy

# Load the spaCy model
nlp = spacy.load("la_core_web_lg")

# Get the model version
model_version = nlp.meta["version"]

# Get spaCy version
print(f"The spaCy version is: {spacy.__version__}")

# Print the spaCy model name
print(f"The spaCy model name is: {nlp.meta['name']}")

# Print the model version
print(f"The spaCy model version is: {model_version}")

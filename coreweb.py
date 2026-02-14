import spacy

# Load the pre-trained spaCy model
# This model is a general-purpose English model, suitable for demonstrating NER.
nlp = spacy.load("en_core_web_md")

# Sample clinical text for entity extraction
clinical_text = "The patient presented with chest pain, fever, and a persistent cough. Diagnosis included pneumonia and possible bronchitis. Prescribed amoxicillin 500mg daily."

# Process the clinical text with the spaCy model
doc = nlp(clinical_text)

# Extract and display entities and their labels
print("Extracted Medical Entities:")
if doc.ents:
    for ent in doc.ents:
        print(f"  - Entity: {ent.text}, Type: {ent.label_}")
else:
    print("  No entities found.")

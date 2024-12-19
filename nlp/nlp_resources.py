import spacy

def get_nlp_instance():
    return spacy.load('en_core_web_sm')
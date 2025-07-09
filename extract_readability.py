import pandas as pd
import numpy as np
import spacy
import readability

# Chargement du modèle spaCy pour le français
nlp = spacy.load("fr_core_news_sm")

# Extraction des mesures de lisibilité
def get_features(text, lang='fr'):
    nlp = get_nlp()
    doc = nlp(text)
    
    # Reformater le texte
    tokenized = '\n\n'.join(' '.join(token.text for token in sent) for sent in doc.sents)
    results = readability.getmeasures(tokenized, lang=lang, merge=True)

    # Supprimer les mesures qu'on n'utilise pas
    for key in [
        'Kincaid', 'ARI', 'Coleman-Liau', 'FleschReadingEase',
        'GunningFogIndex', 'SMOGIndex', 'DaleChallIndex',
        'paragraphs', 'complex_words_dc'
    ]:
        results.pop(key, None)

    return results

# Calcul des différences de lisibilité entre deux phrases
def extract_readability_features(original, simplified):
    ori_feats = get_features(original)
    sim_feats = get_features(simplified)

    # Convertir les résultats en Series pandas
    ori_df = pd.Series(ori_feats)
    sim_df = pd.Series(sim_feats)

    # Calcul de la différence (simplifiée - originale)
    diff_df = sim_df - ori_df
    diff_df = diff_df.add_prefix("diff_")

    return pd.DataFrame([diff_df])

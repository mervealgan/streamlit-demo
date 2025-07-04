import pandas as pd
import numpy as np
import spacy
import readability

nlp = spacy.load("fr_core_news_sm")

def get_features(text, lang='fr'):
    doc = nlp(text)
    tokenized = '\n\n'.join(' '.join(token.text for token in sent) for sent in doc.sents)
    results = readability.getmeasures(tokenized, lang=lang, merge=True)

    for key in [
        'Kincaid', 'ARI', 'Coleman-Liau', 'FleschReadingEase',
        'GunningFogIndex', 'SMOGIndex', 'DaleChallIndex',
        'paragraphs', 'complex_words_dc'
    ]:
        results.pop(key, None)

    return results

def extract_readability_features(original, simplified):
    ori_feats = get_features(original)
    sim_feats = get_features(simplified)

    ori_df = pd.Series(ori_feats).add_prefix("ori_")
    sim_df = pd.Series(sim_feats).add_prefix("sim_")
    diff_df = sim_df - ori_df
    diff_df = diff_df.add_prefix("diff_")

    return pd.DataFrame([diff_df])

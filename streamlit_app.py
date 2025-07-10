import streamlit as st
import pandas as pd
import joblib

from extract_readability import extract_readability_features
from extract_plongements_camembert import extract_camembert_diff

import spacy

FEATURE_LABELS = {
    "diff_LIX": "Δ LIX",
    "diff_RIX": "Δ RIX",
    "diff_REL": "Δ REL",
    "diff_KandelMoles": "Δ Kandel-Moles",
    "diff_Mesnager": "Δ Mesnager",
    "diff_characters_per_word": "Δ caractères/mot",
    "diff_syll_per_word": "Δ syllabes/mot",
    "diff_words_per_sentence": "Δ mots/phrase",
    "diff_sentences_per_paragraph": "Δ phrases/paragraphe",
    "diff_type_token_ratio": "Δ type_token_ratio",
    "diff_directspeech_ratio": "Δ proportion de discours direct",
    "diff_characters": "Δ nombre de caractères",
    "diff_syllables": "Δ nombre de syllabes",
    "diff_words": "Δ nombre de mots",
    "diff_wordtypes": "Δ nombre de mots différents",
    "diff_sentences": "Δ nombre de phrases",
    "diff_long_words": "Δ mots longs",
    "diff_complex_words": "Δ mots complexes",
    "diff_complex_words_mes": "Δ mots complexes (Mesnager)",
    "diff_tobeverb": "Δ verbes être",
    "diff_auxverb": "Δ verbes auxiliaires",
    "diff_conjunction": "Δ conjonctions",
    "diff_preposition": "Δ prépositions",
    "diff_nominalization": "Δ nominalisations",
    "diff_subordination": "Δ subordonnées",
    "diff_article": "Δ articles",
    "diff_pronoun": "Δ pronoms",
    "diff_interrogative": "Δ mots interrogatifs",
}

try:
    nlp = spacy.load("fr_core_news_sm")
except:
    import spacy.cli
    spacy.cli.download("fr_core_news_sm")
    nlp = spacy.load("fr_core_news_sm")

# Load models
model = joblib.load("mlp_exp_max_rev_read_model.pkl")
pca = joblib.load("pca_model_max_rev.pkl")

# App layout
st.title("Prédiction de l’amélioration de lisibilité")
st.write("Entrez une phrase **originale** et sa version **simplifiée**.")

original = st.text_area("Phrase originale")
simplified = st.text_area("Phrase simplifiée")

if st.button("Prédire"):
    if original.strip() == simplified.strip():
        value = 0.0
        features = pd.DataFrame()
    else:
        emb_df = extract_camembert_diff(original, simplified)
        emb_pca = pd.DataFrame(pca.transform(emb_df), columns=[f"pca_{i+1}" for i in range(pca.n_components_)])
        read_df = extract_readability_features(original, simplified)
        features = pd.concat([emb_pca, read_df], axis=1)
        value = model.predict(features)[0]

    st.subheader(f"Score prédit : {round(value, 2)}")
    st.markdown(
        f"""
        <div style="width: 100%; height: 25px; background: linear-gradient(to right, red, gray, green); position: relative; border-radius: 5px; margin-top: 10px;">
            <div style="position: absolute; left: {(value + 3) / 6 * 100}%; top: -5px; width: 0; height: 0;
                        border-left: 7px solid transparent; border-right: 7px solid transparent;
                        border-bottom: 10px solid black;"></div>
        </div>
        <div style="text-align: center; font-size: 14px; margin-top: 5px;">
            Échelle : -3 (plus difficile) → +3 (plus facile)
        </div>
        """,
        unsafe_allow_html=True
    )

    readable_features_only = features[[col for col in features.columns if not col.startswith("pca_")]]
    st.dataframe(readable_features_only.round(3))

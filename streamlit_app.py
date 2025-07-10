import streamlit as st
import pandas as pd
import joblib
from extract_readability import extract_readability_features
from extract_plongements_camembert import extract_camembert_diff
import spacy

# Dictionnaire pour rendre les noms de caractéristiques plus lisibles avec explications
FEATURE_LABELS = {
    "diff_LIX": "Indice LIX (complexité générale)",
    "diff_RIX": "Indice RIX (mots difficiles)",
    "diff_REL": "Indice REL (longueur relative)",
    "diff_KandelMoles": "Indice Kandel-Moles",
    "diff_Mesnager": "Indice Mesnager",
    "diff_characters_per_word": "Caractères par mot",
    "diff_syll_per_word": "Syllabes par mot",
    "diff_words_per_sentence": "Mots par phrase",
    "diff_sentences_per_paragraph": "Phrases par paragraphe",
    "diff_type_token_ratio": "Richesse du vocabulaire",
    "diff_directspeech_ratio": "Proportion de discours direct",
    "diff_characters": "Nombre total de caractères",
    "diff_syllables": "Nombre total de syllabes",
    "diff_words": "Nombre total de mots",
    "diff_wordtypes": "Nombre de mots différents",
    "diff_sentences": "Nombre total de phrases",
    "diff_long_words": "Mots longs (+ de 6 lettres)",
    "diff_complex_words": "Mots complexes (+ de 2 syllabes)",
    "diff_complex_words_mes": "Mots complexes (méthode Mesnager)",
    "diff_tobeverb": "Verbes 'être'",
    "diff_auxverb": "Verbes auxiliaires",
    "diff_conjunction": "Conjonctions",
    "diff_preposition": "Prépositions",
    "diff_nominalization": "Nominalisations",
    "diff_subordination": "Propositions subordonnées",
    "diff_article": "Articles",
    "diff_pronoun": "Pronoms",
    "diff_interrogative": "Mots interrogatifs",
}

# Explications détaillées pour les utilisateurs
FEATURE_EXPLANATIONS = {
    "Indice LIX (complexité générale)": "Mesure globale de difficulté de lecture basée sur la longueur des mots et des phrases",
    "Indice RIX (mots difficiles)": "Proportion de mots longs et difficiles dans le texte",
    "Indice REL (longueur relative)": "Mesure de la longueur relative des phrases et mots",
    "Caractères par mot": "Longueur moyenne des mots (moins = plus simple)",
    "Syllabes par mot": "Complexité phonétique moyenne (moins = plus simple)",
    "Mots par phrase": "Longueur des phrases (moins = plus simple)",
    "Richesse du vocabulaire": "Diversité des mots utilisés",
    "Proportion de discours direct": "Utilisation de dialogue (plus = plus vivant)",
    "Mots longs (+ de 6 lettres)": "Nombre de mots potentiellement difficiles",
    "Mots complexes (+ de 2 syllabes)": "Mots phonétiquement complexes",
    "Verbes 'être'": "Usage du verbe être (simplicité syntaxique)",
    "Conjonctions": "Mots de liaison (cohérence du texte)",
    "Prépositions": "Mots de relation spatiale/temporelle",
    "Nominalisations": "Transformation de verbes en noms (moins = plus simple)",
    "Propositions subordonnées": "Phrases complexes imbriquées (moins = plus simple)",
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
st.title("Prédiction de l'amélioration de lisibilité")
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
    
    # Filter out PCA features and rename columns with readable labels
    readable_features_only = features[[col for col in features.columns if not col.startswith("pca_")]]
    
    if not readable_features_only.empty:
        # Create a copy and rename columns using the feature labels
        display_features = readable_features_only.copy()
        display_features = display_features.rename(columns=FEATURE_LABELS)
        
        st.subheader("Analyse détaillée des changements de lisibilité")
        
        # Add explanation
        st.info("Comment lire ce tableau :\n"
                "- **Valeurs négatives** (-) = la version simplifiée a MOINS de cette caractéristique → plus facile à lire\n"
                "- **Valeurs positives** (+) = la version simplifiée a PLUS de cette caractéristique → peut être plus difficile\n"
                "- **Zéro** (0) = pas de changement")
        
        # Show the dataframe
        st.dataframe(display_features.round(3), use_container_width=True)
        
        # Add collapsible explanations
        with st.expander("Que signifient ces mesures ?"):
            st.write("**Cliquez sur une mesure pour en savoir plus :**")
            
            # Create columns for better layout
            col1, col2 = st.columns(2)
            
            with col1:
                for feature, explanation in list(FEATURE_EXPLANATIONS.items())[:len(FEATURE_EXPLANATIONS)//2]:
                    if feature in display_features.columns:
                        st.write(f"**{feature}** : {explanation}")
            
            with col2:
                for feature, explanation in list(FEATURE_EXPLANATIONS.items())[len(FEATURE_EXPLANATIONS)//2:]:
                    if feature in display_features.columns:
                        st.write(f"**{feature}** : {explanation}")
    else:
        st.write("Aucune différence détectée entre les phrases.")

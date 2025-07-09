import streamlit as st
import pandas as pd
import joblib

from extract_readability import extract_readability_features
from extract_plongements_camembert import extract_camembert_diff

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

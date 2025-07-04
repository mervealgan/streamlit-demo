import streamlit as st
import joblib
import pandas as pd
from extract_readability import extract_readability_features
from extraction_plongements_camembert import extract_camembert_diff

# Page config
st.set_page_config(
    page_title="Prédiction de Lisibilité",
    page_icon="📚",
    layout="wide"
)

# Cache the model loading
@st.cache_resource
def load_model():
    """Load the trained model."""
    try:
        model = joblib.load("random_forest_exp_max_read_model.pkl")
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        return None

# Load model
model = load_model()

# Title and description
st.title("📚 Prédiction de Lisibilité après Simplification")
st.markdown("""
Cette application prédit le gain de lisibilité obtenu en simplifiant un texte.
Entrez votre phrase originale et sa version simplifiée pour obtenir une estimation.
""")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("Phrase originale")
    original = st.text_area(
        "Entrez la phrase originale ici:",
        height=100,
        placeholder="Tapez votre texte original...",
        key="original"
    )

with col2:
    st.subheader("Phrase simplifiée")
    simplified = st.text_area(
        "Entrez la phrase simplifiée ici:",
        height=100,
        placeholder="Tapez votre texte simplifié...",
        key="simplified"
    )

# Predict button
if st.button("🔍 Prédire le gain de lisibilité", type="primary"):
    if not original or not simplified:
        st.error("⚠️ Veuillez entrer du texte dans les deux champs.")
    elif not model:
        st.error("❌ Le modèle n'a pas pu être chargé.")
    else:
        try:
            with st.spinner("Analyse en cours..."):
                # Extract features
                emb_df = extract_camembert_diff(original, simplified)
                read_df = extract_readability_features(original, simplified)
                
                # Combine features
                full_input = pd.concat([emb_df, read_df], axis=1)
                
                # Make prediction
                pred = model.predict(full_input)[0]
                
                # Display result
                st.success(f"✅ **Gain prédit: {round(pred, 2)}**")
                
                # Additional info
                st.info(f"""
                📊 **Statistiques:**
                - Longueur originale: {len(original)} caractères
                - Longueur simplifiée: {len(simplified)} caractères
                - Réduction: {len(original) - len(simplified)} caractères
                """)
                
        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction: {str(e)}")

# Examples section
st.markdown("---")
st.subheader("📝 Exemples")

example_col1, example_col2 = st.columns(2)

with example_col1:
    if st.button("Exemple 1"):
        st.session_state.original = "Cette méthodologie complexe nécessite une compréhension approfondie des paradigmes théoriques."
        st.session_state.simplified = "Cette méthode demande de bien comprendre les idées de base."
        st.rerun()

with example_col2:
    if st.button("Exemple 2"):
        st.session_state.original = "L'implémentation de cette fonctionnalité requiert une expertise technique considérable."
        st.session_state.simplified = "Ajouter cette fonction demande beaucoup de connaissances techniques."
        st.rerun()

# Footer
st.markdown("---")
st.markdown("*Développé avec Streamlit*")

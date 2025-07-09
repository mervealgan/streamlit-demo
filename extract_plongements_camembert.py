import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

# Chargement du modèle CamemBERT
model_name = "camembert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

# Utilisation du GPU si disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Fonction de max pooling sur les embeddings de tokens
def max_pooling(token_embeddings, attention_mask):
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
    token_embeddings[mask_expanded == 0] = -1e9
    return torch.max(token_embeddings, dim=1)[0]

# Extraction d'un vecteur de phrase avec max pooling
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        pooled = max_pooling(token_embeddings, attention_mask)
    return pooled.squeeze().cpu().numpy()

# Calcul de la différence d'embedding entre deux phrases
def extract_camembert_diff(original, simplified):
    ori_vec = get_embedding(original)
    sim_vec = get_embedding(simplified)
    diff_vec = sim_vec - ori_vec
    return pd.DataFrame([diff_vec], columns=[f"max_{i}" for i in range(len(diff_vec))])
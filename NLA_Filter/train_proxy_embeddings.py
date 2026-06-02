import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, average_precision_score

# ==========================================
# CONFIGURATION
# ==========================================
TRAINING_FILE = "nla_45k_training_data.jsonl"
MODEL_OUTPUT = "nla_45k_proxy_embeddings.lgb"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
BATCH_SIZE = 512  # Your A5000s will chew through this instantly

# Load the embedding model onto the GPU
print(f"Loading {EMBEDDING_MODEL} onto GPU...")
embedder = SentenceTransformer(EMBEDDING_MODEL, device='cuda')

# ==========================================
# 1. BATCH FEATURE EXTRACTION
# ==========================================
def extract_features_batch(texts):
    # 1. Get the 384-dimension semantic vectors (GPU Accelerated)
    embeddings = embedder.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False)
    
    # 2. Append basic structural heuristics (Length & Word Count)
    features = []
    for i, text in enumerate(texts):
        words = text.split()
        char_len = len(text)
        word_count = len(words)
        
        # Combine the 384 semantic features with the 2 structural features
        row_features = np.append(embeddings[i], [char_len, word_count])
        features.append(row_features)
        
    return np.array(features)

# ==========================================
# 2. TRAINING PIPELINE
# ==========================================
def train_model():
    print("Reading JSONL data...")
    texts = []
    labels = []
    
    with open(TRAINING_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            texts.append(data['text'])
            labels.append(data['is_golden'])
            
    print(f"Generating Semantic Embeddings for {len(texts)} examples...")
    X = extract_features_batch(texts)
    y = np.array(labels)
    
    # Convert to DataFrame (LightGBM prefers this for tracking features)
    feature_names = [f"emb_{i}" for i in range(384)] + ["char_len", "word_count"]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    print(f"Data ready. Shape: {X_df.shape}. Golden ratio: {y.mean():.2%}")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    models = []
    fold_aucs = []
    
    print("\nStarting 5-Fold Cross Validation...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_df, y)):
        X_tr, X_val = X_df.iloc[train_idx], X_df.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        model = lgb.LGBMClassifier(
            n_estimators=700,         # Slightly higher for 386 features
            learning_rate=0.03,       # Slower learning for deeper semantic patterns
            num_leaves=31,
            min_child_samples=15,
            class_weight="balanced",
            subsample=0.8,
            colsample_bytree=0.8,     # Randomly sample features to prevent overfitting
            n_jobs=-1,
            random_state=42,
            verbosity=-1
        )
        
        callbacks = [lgb.early_stopping(50, verbose=False)]
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=callbacks)
        
        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds
        models.append(model)
        
        fold_auc = roc_auc_score(y_val, val_preds)
        fold_aucs.append(fold_auc)
        print(f"Fold {fold+1} AUC: {fold_auc:.4f}")

    # ==========================================
    # EVALUATION & METRICS
    # ==========================================
    print("\n=== OVERALL CROSS-VALIDATION RESULTS ===")
    overall_auc = roc_auc_score(y, oof_preds)
    pr_auc = average_precision_score(y, oof_preds)
    
    print(f"Overall ROC AUC: {overall_auc:.4f}")
    print(f"PR AUC (Avg Precision): {pr_auc:.4f}")
    
    hard_preds = (oof_preds >= 0.5).astype(int)
    print("\nConfusion Matrix (At 0.5 Threshold):")
    print(confusion_matrix(y, hard_preds))
    print("\nClassification Report (Precision/Recall/F1 at 0.5):")
    print(classification_report(y, hard_preds))

    # Save the best model
    best_idx = np.argmax(fold_aucs)
    best_model = models[best_idx]
    best_model.booster_.save_model(MODEL_OUTPUT)
    print(f"\nSaved High-Purity Proxy to {MODEL_OUTPUT}")

# ==========================================
# 3. LIVE TESTING FUNCTION
# ==========================================
def test_custom_text(text_string):
    bst = lgb.Booster(model_file=MODEL_OUTPUT)
    
    # We must format it as a list of 1 to pass to the batch extractor
    X_test = extract_features_batch([text_string])
    
    prob = bst.predict(X_test)[0]
    verdict = 1 if prob >= 0.5 else 0
    
    print("\n--- TEST RESULT ---")
    print(f"Input Text: {text_string[:100]}...")
    print(f"Raw Probability Score: {prob:.4f}")
    
    if verdict == 1:
        if prob >= 0.9: print("Verdict: KEEP (Tier 1 - Confidently Golden)")
        elif prob >= 0.7: print("Verdict: KEEP (Tier 2 - Strong Logic)")
        else: print("Verdict: KEEP (Tier 3 - Borderline Golden)")
    else:
        print("Verdict: DROP (Fluff / Low Logic)")

if __name__ == "__main__":
    train_model()
    
    test_custom_text("Hey, how are you doing today? I was wondering if you could tell me a story about a brave knight.")
    
    test_custom_text("""
    Given that the base angles of the isosceles triangle are equal, it follows that the sides opposite 
    those angles must also be equal. Therefore, step 1 implies that line segment AB equals line segment AC. 
    Because of this geometric property, we can thus conclude the proof.
    """)
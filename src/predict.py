"""
Inference and Prediction Module for Customer Review Sentiment Analysis.
Supports both:
1. Fast Local TF-IDF + Calibrated Classifier Pipeline
2. Hugging Face DistilBERT Deep Learning Transformer
Includes feature attribution / keyword explainability for review insights.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import clean_text

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PIPELINE_PATH = os.path.join(MODELS_DIR, "sentiment_tfidf_pipeline.pkl")

# Cached instances
_TFIDF_PIPELINE = None
_TRANSFORMER_PIPELINE = None

def get_tfidf_pipeline():
    """Load or return cached TF-IDF + Classifier pipeline."""
    global _TFIDF_PIPELINE
    if _TFIDF_PIPELINE is None:
        if not os.path.exists(PIPELINE_PATH):
            raise FileNotFoundError(
                f"Trained model pipeline not found at {PIPELINE_PATH}. "
                "Please run `python src/train.py` first to train and export the model."
            )
        _TFIDF_PIPELINE = joblib.load(PIPELINE_PATH)
    return _TFIDF_PIPELINE

def get_transformer_pipeline():
    """Load or return cached Hugging Face DistilBERT sentiment pipeline."""
    global _TRANSFORMER_PIPELINE
    if _TRANSFORMER_PIPELINE is None:
        from transformers import pipeline
        _TRANSFORMER_PIPELINE = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
    return _TRANSFORMER_PIPELINE

def explain_prediction(text: str, pipeline=None, top_k: int = 5):
    """
    Extract top positive and negative influential tokens driving the sentiment decision
    based on TF-IDF token weights and classifier coefficients.
    """
    if pipeline is None:
        pipeline = get_tfidf_pipeline()

    tfidf_vec = pipeline.named_steps.get("tfidf")
    clf = pipeline.named_steps.get("classifier")

    if not tfidf_vec or not clf:
        return []

    # Get coefficients from model
    if hasattr(clf, "coef_"):
        coefs = clf.coef_[0]
    elif hasattr(clf, "calibrated_classifiers_"):
        # Average coefficients across calibrated base estimators
        coef_list = [c.estimator.coef_[0] for c in clf.calibrated_classifiers_ if hasattr(c.estimator, "coef_")]
        if coef_list:
            coefs = np.mean(coef_list, axis=0)
        else:
            return []
    else:
        return []

    cleaned = clean_text(text)
    if not cleaned:
        return []

    # Vectorize single text
    x_vec = tfidf_vec.transform([text])
    feature_names = tfidf_vec.get_feature_names_out()

    non_zero_indices = x_vec.nonzero()[1]
    if len(non_zero_indices) == 0:
        return []

    scores = []
    for idx in non_zero_indices:
        token = feature_names[idx]
        tfidf_val = x_vec[0, idx]
        weight = coefs[idx] * tfidf_val
        scores.append({
            "word": token,
            "score": float(weight),
            "sentiment": "Positive" if weight > 0 else "Negative",
            "abs_impact": abs(weight)
        })

    # Sort by absolute impact
    scores = sorted(scores, key=lambda x: x["abs_impact"], reverse=True)
    return scores[:top_k]

def predict_single_tfidf(text: str) -> dict:
    """Predict sentiment using the custom trained TF-IDF + ML Pipeline."""
    pipeline = get_tfidf_pipeline()
    
    # Compute probability
    proba = pipeline.predict_proba([text])[0]
    neg_p, pos_p = float(proba[0]), float(proba[1])
    
    pred_label = int(pipeline.predict([text])[0])
    sentiment = "Positive" if pred_label == 1 else "Negative"
    confidence = pos_p if pred_label == 1 else neg_p

    keywords = explain_prediction(text, pipeline=pipeline, top_k=5)

    return {
        "text": text,
        "sentiment": sentiment,
        "confidence": round(confidence, 4),
        "probabilities": {
            "Positive": round(pos_p, 4),
            "Negative": round(neg_p, 4)
        },
        "model_used": "Custom TF-IDF + ML Pipeline (Local)",
        "word_count": len(text.split()),
        "keywords": keywords
    }

def predict_single_transformer(text: str) -> dict:
    """Predict sentiment using Hugging Face DistilBERT Transformer."""
    hf_pipe = get_transformer_pipeline()
    
    # DistilBERT has a 512 max token limit; truncate cleanly
    truncated_text = text[:1500]
    res = hf_pipe(truncated_text)[0]
    
    label_raw = res["label"].upper()
    score = float(res["score"])
    
    if label_raw == "POSITIVE":
        sentiment = "Positive"
        pos_p = score
        neg_p = 1.0 - score
    else:
        sentiment = "Negative"
        neg_p = score
        pos_p = 1.0 - score

    return {
        "text": text,
        "sentiment": sentiment,
        "confidence": round(score, 4),
        "probabilities": {
            "Positive": round(pos_p, 4),
            "Negative": round(neg_p, 4)
        },
        "model_used": "Hugging Face DistilBERT Transformer",
        "word_count": len(text.split()),
        "keywords": []
    }

def predict_sentiment(text: str, model_type: str = "tfidf") -> dict:
    """Unified entry point for sentiment prediction."""
    if not text or not text.strip():
        return {
            "text": "",
            "sentiment": "Neutral",
            "confidence": 0.0,
            "probabilities": {"Positive": 0.5, "Negative": 0.5},
            "model_used": model_type,
            "word_count": 0,
            "keywords": []
        }

    if model_type.lower() in ["transformer", "distilbert"]:
        return predict_single_transformer(text)
    return predict_single_tfidf(text)

def predict_batch(df: pd.DataFrame, text_column: str, model_type: str = "tfidf") -> pd.DataFrame:
    """
    Perform high-speed batch sentiment classification on a Pandas DataFrame.
    """
    df_out = df.copy()
    texts = df_out[text_column].astype(str).fillna("").tolist()

    if model_type.lower() in ["transformer", "distilbert"]:
        hf_pipe = get_transformer_pipeline()
        preds = []
        confs = []
        # Batch inference with HuggingFace
        for t in texts:
            truncated = t[:1500] if len(t) > 1500 else t
            res = hf_pipe(truncated)[0]
            label = "Positive" if res["label"].upper() == "POSITIVE" else "Negative"
            preds.append(label)
            confs.append(round(float(res["score"]), 4))
        df_out["predicted_sentiment"] = preds
        df_out["confidence"] = confs
    else:
        pipeline = get_tfidf_pipeline()
        probas = pipeline.predict_proba(texts)
        y_preds = pipeline.predict(texts)
        
        sentiments = ["Positive" if p == 1 else "Negative" for p in y_preds]
        confs = [round(float(probas[i, p]), 4) for i, p in enumerate(y_preds)]
        
        df_out["predicted_sentiment"] = sentiments
        df_out["confidence"] = confs

    return df_out

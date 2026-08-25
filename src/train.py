##this contain the complete training pipeline

import joblib
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from preprocessing import clean_text


# ==========================================
# 1. Project Paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "spam.csv"
MODEL_PATH = BASE_DIR / "models" / "spam_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"


# ==========================================
# 2. Load Dataset
# ==========================================

print("Loading dataset...")

df = pd.read_csv(
    DATA_PATH,
    encoding="latin-1"
)


# ==========================================
# 3. Select Required Columns
# ==========================================

df = df[['v1', 'v2']]

df.columns = ['class', 'sms']


# ==========================================
# 4. Remove Duplicates
# ==========================================

df = df.drop_duplicates(
    keep='first'
)

df = df.reset_index(
    drop=True
)


print(
    f"Dataset size after cleaning: {df.shape}"
)


# ==========================================
# 5. Separate Features and Labels
# ==========================================

X = df['sms']

y = df['class']


# ==========================================
# 6. Train/Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Testing samples: {len(X_test)}"
)


# ==========================================
# 7. Text Preprocessing
# ==========================================

print("Preprocessing text...")

X_train_clean = X_train.apply(
    clean_text
)

X_test_clean = X_test.apply(
    clean_text
)


# ==========================================
# 8. TF-IDF Feature Extraction
# ==========================================

print("Creating TF-IDF features...")

tfidf = TfidfVectorizer(
    max_features=3000
)

X_train_tfidf = tfidf.fit_transform(
    X_train_clean
)

X_test_tfidf = tfidf.transform(
    X_test_clean
)


# ==========================================
# 9. Train Linear SVM
# ==========================================

print("Training Linear SVM...")

model = LinearSVC()

model.fit(
    X_train_tfidf,
    y_train
)


# ==========================================
# 10. Make Predictions
# ==========================================

predictions = model.predict(
    X_test_tfidf
)


# ==========================================
# 11. Evaluate Model
# ==========================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    pos_label='spam'
)

recall = recall_score(
    y_test,
    predictions,
    pos_label='spam'
)

f1 = f1_score(
    y_test,
    predictions,
    pos_label='spam'
)


print("\n===================================")
print("MODEL PERFORMANCE")
print("===================================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


print("\nClassification Report")
print("-----------------------------------")

print(
    classification_report(
        y_test,
        predictions
    )
)


# ==========================================
# 12. Save Model
# ==========================================

print("Saving model...")

joblib.dump(
    model,
    MODEL_PATH
)


# ==========================================
# 13. Save TF-IDF Vectorizer
# ==========================================

print("Saving TF-IDF vectorizer...")

joblib.dump(
    tfidf,
    VECTORIZER_PATH
)


print("\n===================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("===================================")

print(
    f"Model saved to: {MODEL_PATH}"
)

print(
    f"Vectorizer saved to: {VECTORIZER_PATH}"
)

##this 'train' file contain only the steps required to train the final model.


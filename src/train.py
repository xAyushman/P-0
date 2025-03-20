import pandas as pd
import joblib
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, precision_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV

# Load the dataset
df = pd.read_csv('../data/spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=8000, min_df=2)
X_tfidf = vectorizer.fit_transform(df['message'])
y = df['label']

# Apply SMOTE for balancing the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_tfidf, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Hyperparameter tuning for alpha
param_grid = {'alpha': [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]}
grid_search = GridSearchCV(MultinomialNB(), param_grid, scoring='accuracy', cv=5)
grid_search.fit(X_train, y_train)

# Best alpha selection
best_alpha = grid_search.best_params_['alpha']
print(f'Best alpha: {best_alpha}')

# Train Naive Bayes Model with best alpha
model = MultinomialNB(alpha=best_alpha)
model.fit(X_train, y_train)

# Adjust Decision Threshold
def custom_predict(model, X, threshold=0.85):
    probabilities = model.predict_proba(X)[:, 1]  # Get probability of spam
    return (probabilities >= threshold).astype(int)

# Predictions with custom threshold
y_pred = custom_predict(model, X_test, threshold=0.8)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, '../models/spam_classifier.pkl')
joblib.dump(vectorizer, '../models/tfidf_vectorizer.pkl')


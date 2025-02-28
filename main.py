import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Ensure vectorizer is always initialized
vectorizer = TfidfVectorizer(max_features=5000)  #  Move it outside the try block

try:
    df = pd.read_csv('processed_emails.csv')
    X_df = pd.read_csv('tfidf_features.csv')
    print("Loaded preprocessed data successfully.")

    #  Fit vectorizer even when loading preprocessed data
    df['Cleaned Email'] = df['Cleaned Email'].fillna("")  #  Replace NaN with an empty string
    X = vectorizer.fit_transform(df['Cleaned Email'])
    X_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

except FileNotFoundError:
    df = pd.read_csv('email.csv')
    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
        return ' '.join(words)

    df['Cleaned Email'] = df['Message'].apply(preprocess_text)

    print(df[['Message', 'Cleaned Email']].head())

    #  Fit vectorizer only if preprocessing is done
    X = vectorizer.fit_transform(df['Cleaned Email'])
    X_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

    df.to_csv('processed_emails.csv', index=False)
    X_df.to_csv('tfidf_features.csv', index=False)
    print("Saved preprocessed data and TF-IDF features.")

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(df['Category'])

print(y[:10])

X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

#  Save both model and vectorizer after ensuring vectorizer exists
with open('spam_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved.")

import pandas as pd
import numpy as np
import string
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Loading the dataset
df = pd.read_csv("D:\Internship projects/spam.csv", encoding='latin-1')

# Displaying the first few rows
print("First few rows:")
print(df.head())

# Droping unnecessary columns (only keeping 'v1' and 'v2')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Maping labels to binary values
df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

# Checking for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Data visualization 
sns.countplot(data=df, x='label')
plt.title('Distribution of Spam vs Ham')
plt.show()

# Text preprocessing
def clean_text(msg):
    msg = msg.lower()
    msg = ''.join([char for char in msg if char not in string.punctuation])
    return msg

df['clean_msg'] = df['message'].apply(clean_text)

# Feature extracting using CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(df['clean_msg'])

# Labeling
y = df['label_num']

# Spliting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training with Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Predicting
y_pred = model.predict(X_test)

# Evaluating
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Testing the model on new examples
def predict_message(msg):
    msg_clean = clean_text(msg)
    msg_vector = cv.transform([msg_clean])
    prediction = model.predict(msg_vector)
    return "Spam" if prediction[0] == 1 else "Ham"

# Example usage
sample = "Congratulations! You've won a $1000 gift card. Click here to claim now."
print(f"\nSample prediction: '{sample}' → {predict_message(sample)}")

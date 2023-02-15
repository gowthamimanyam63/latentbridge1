import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 

# Load the dataset into a pandas dataframe
data = pd.read_csv('dataset.csv')

# Preprocess the data by removing stopwords and converting the text data into numerical form
stop_words = set(stopwords.words('english'))
vectorizer = CountVectorizer(stop_words=stop_words)
X = vectorizer.fit_transform(data['comment'].values.astype('U'))
y = data['sentiment']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier on the training data
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Use the trained model to make predictions on the testing data
y_pred = clf.predict(X_test)

# Evaluate the performance of the model using accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy) 
# Importing necessary libraries
import re
import nltk
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer

# Loading the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# Downloading the stopwords from NLTK
nltk.download('stopwords')

# Preprocessing the reviews
corpus = []
for i in range(0, len(dataset)):
    # Removing non-alphabetic characters
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    # Converting to lower case
    review = review.lower()
    # Tokenization
    review = review.split()
    # Stemming and removing stopwords
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')  # Removing 'not' from stopwords as it's important for sentiment analysis
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    # Joining the words back into a string
    review = ' '.join(review)
    # Adding the processed review to the corpus
    corpus.append(review)

# Vectorizing the text data
cv = CountVectorizer(max_features=1550)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values  # Labels

# Splitting the dataset into Training and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Creating and training the Support Vector Classifier
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Comparing predicted and actual values
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis=1))

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix, accuracy)

# New Review Entry

new_review = []
new_review = input("Type review: ")

# Cleaning the text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
review = re.sub('[^a-zA-Z]', ' ',new_review)
review = review.lower()
review = review.split()
review = [word for word in review if not word in stopwords.words('english')]
ps = PorterStemmer()
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
review = ' '.join(review)
corpus.append(review)

print(corpus)

# We assume new reviews will be of similar form to the dataset, so not need scaling

# Running the model on the input
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
cv = joblib.load('nlp_cv.pkl')
x = cv.transform(corpus).toarray()
classifier = joblib.load('nlp_model.pkl')
y = classifier.predict(x)
print('We got a result! ', y)

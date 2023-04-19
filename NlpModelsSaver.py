import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle


def create_nlp_features(text, value=None):
    #nltk.download('vader_lexicon')
    #nltk.download('punkt')
    #nltk.download('averaged_perceptron_tagger')

    # Используем SentimentIntensityAnalyzer для определения тональности текста
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    tokens = word_tokenize(text)
    # POS-теггинг для определения частей речи
    pos_tags = nltk.pos_tag(tokens)
    joined_tokens = ' '.join(tokens)
    features = [
        sentiment['pos'],
        sentiment['neg'],
        sentiment['neu'],
        len(tokens),
        len(set(tokens)),
        len(nltk.FreqDist(tokens)),
        len(nltk.FreqDist(pos_tags)),
        len(nltk.FreqDist(joined_tokens))
        ]
    if (value is not None):
        features.append(value)
    return features

regular_lines = [line.strip() for line in open('regularContents.txt', 'r')]
regular_lines_features = [create_nlp_features(item, 0) for item in regular_lines]

phishing_lines = [line.strip() for line in open('phishingContents.txt', 'r')]
phishing_lines_features = [create_nlp_features(item, 1) for item in phishing_lines]

features = []

for x, y in zip(regular_lines_features, phishing_lines_features):
    features.append(x)
    features.append(y)

df = pd.DataFrame(features)

x = df.iloc[:, :8].values.tolist()
y = df[8].values.tolist()
train_set_x = x[:30]
train_set_y = y[:30]

test_set_x = x[30:]
test_set_y = y[30:]

svmclf = svm.SVC(kernel="rbf")
svmclf.fit(train_set_x, train_set_y)
with open('svm_nlp.pkl', 'wb') as f:
    pickle.dump(svmclf, f)

clf_neur = MLPClassifier(activation = "logistic", solver='lbfgs', max_iter = 500, hidden_layer_sizes = (100), random_state=1)
clf_neur.fit(train_set_x, train_set_y)
with open('neur_nlp.pkl', 'wb') as f:
    pickle.dump(clf_neur, f)

desclf = DecisionTreeClassifier(max_depth=1)
desclf.fit(train_set_x, train_set_y)
with open('decision_nlp.pkl', 'wb') as f:
    pickle.dump(desclf, f)

kneighbclf = KNeighborsClassifier(n_neighbors=13)
kneighbclf.fit(train_set_x, train_set_y)
with open('kneighb_nlp.pkl', 'wb') as f:
    pickle.dump(kneighbclf, f)

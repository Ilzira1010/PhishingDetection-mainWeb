from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle

df = pd.read_csv(r'phishing-dataset-version2.csv', delimiter=',')

#df Нормализация столбца time_domain_activation по max/min
df.loc[df['time_domain_activation'] >= 0, 'time_domain_activation'] = (df.loc[df['time_domain_activation'] >= 0, 'time_domain_activation'] - df.loc[df['time_domain_activation'] >= 0, 'time_domain_activation'].min()) / (df.loc[df['time_domain_activation'] >= 0, 'time_domain_activation'].max() - df.loc[df['time_domain_activation'] >= 0, 'time_domain_activation'].min())
x = df.iloc[:, :7].values.tolist()
y = df['phishing'].values.tolist()
train_set_x = x[:2800]
train_set_y = y[:2800]

test_set_x = x[2800:]
test_set_y = y[2800:]

svmclf = svm.SVC(kernel="rbf")
svmclf.fit(train_set_x, train_set_y)
with open('svm.pkl', 'wb') as f:
    pickle.dump(svmclf, f)

clf_neur = MLPClassifier(activation = "logistic", solver='lbfgs', max_iter = 500, hidden_layer_sizes = (100), random_state=1)
clf_neur.fit(train_set_x, train_set_y)
with open('neur.pkl', 'wb') as f:
    pickle.dump(clf_neur, f)

desclf = DecisionTreeClassifier(max_depth=1)
desclf.fit(train_set_x, train_set_y)
with open('decision.pkl', 'wb') as f:
    pickle.dump(desclf, f)

kneighbclf = KNeighborsClassifier(n_neighbors=13)
kneighbclf.fit(train_set_x, train_set_y)
with open('kneighb.pkl', 'wb') as f:
    pickle.dump(kneighbclf, f)
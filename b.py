import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


attributes = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

df = pd.read_csv('data/iris.data', names=attributes)
print(df.head())

X=np.array(df.iloc[:,0:4])
y=np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)

k_values = [i for i in range(1, 50, 2)]

#empty list that will hold cv scores
k_acc_scores = []

for k in k_values:
	knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
	cv_scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
	k_acc_scores.append(cv_scores.mean())

optimal_k = k_values[k_acc_scores.index(max(k_acc_scores))]

print (k_acc_scores)
print("Our optimal k value is {}".format(optimal_k))

plt.plot(k_values, k_acc_scores)
plt.xlabel("K")
plt.ylabel("Acc")
plt.show()

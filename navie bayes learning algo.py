import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.datasets import load_iris
import seaborn as sn
import matplotlib.pyplot as plt

iris = load_iris()
print(iris.feature_names)



X=iris.data
y=iris.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)
GNB=GaussianNB()
GNB.fit(X_train,y_train)
y_pred=GNB.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
confusion_matrix_result=confusion_matrix(y_test,y_pred)
print("Accuracy of the model",accuracy)
print("Confusion matrix of the model:")
print(confusion_matrix_result)
plt.figure(figsize=(7,5))
sn.heatmap(confusion_matrix_result,annot=True)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

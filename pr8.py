Practical No: 08
Aim: Implementing ensemble Learning Algorithm and Support Vector Machine(SVM) Algorithm
1.  Ensemble Learning
import numpy as np
from sklearn import datasets
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
warnings.simplefilter("ignore")
iris = datasets.load_iris()
x,y = iris.data,iris.target
labels = ["Random Forest","Logistic Regression","GaussianNB","Decision Tree"]
m1 = RandomForestClassifier(random_state=42)
m2 = LogisticRegression(random_state=1)
m3 = GaussianNB()
m4 = DecisionTreeClassifier()
for m,label in zip([m1,m2,m3,m4],labels):
    scores = model_selection.cross_val_score(m,x,y,cv=5,scoring="accuracy")
    print(f"Accuracy: {scores.mean()} {label}")
voting_clf_hard = VotingClassifier(estimators=[(labels[0],m1),(labels[1],m2),(labels[2],m3),(labels[3],m4)],voting='hard')
voting_clf_soft = VotingClassifier(estimators=[(labels[0],m1),(labels[1],m2),(labels[2],m3),(labels[3],m4)],voting='soft')
scores1 = model_selection.cross_val_score(voting_clf_hard,x,y,cv=5,scoring="accuracy")
scores2 = model_selection.cross_val_score(voting_clf_soft,x,y,cv=5,scoring="accuracy")
print(f"Accuracy of the hard voting: {scores1.mean()}")
print(f"Accuracy of the soft voting: {scores2.mean()}")

Output: 








2. Support Vector Machine
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # Using only the first two features (sepal length and sepal width)
y = iris.target
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Train the SVM model with a linear kernel
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
# Function to plot decision boundaries
def plot_decision_boundaries(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=100, cmap=plt.cm.Paired)
    plt.xlabel('Sepal length (standardized)')
    plt.ylabel('Sepal width (standardized)')
    plt.title('SVM Decision Boundaries')
    plt.show()
# Plot decision boundaries using the training set
plot_decision_boundaries(X_train, y_train, svm_model)

Output:


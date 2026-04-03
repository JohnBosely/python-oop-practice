from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# Convert to excel
# df.to_excel("titanic.xlsx", index=False)

def load_dataset():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    return pd.read_csv(url)

titanic = load_dataset()

titanic = titanic.drop(["Name", "Embarked", "Cabin", "Fare", "Ticket", "PassengerId"], axis=1)
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
titanic["Sex"] = titanic["Sex"].map({"male": 0, "female": 1})

#Explore the data
# print(df.describe)
# print(df.head())
# print(df.isnull().sum())




X = titanic.drop("Survived", axis=1)
y = titanic["Survived"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


for n, depth in [(100, 5), (200, 5), (100, 10), (200, 10)]:
    rf = RandomForestClassifier(n_estimators=n, max_depth=depth, random_state=42)
    rf.fit(X_train, y_train)
    print(f"Trees: {n} Depth: {depth} Accuracy: {rf.score(X_test, y_test):.4f}")

model = RandomForestClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)
cr = classification_report(y_test, predictions)
print("ACCURACY SCORE: ", accuracy)
print("CONFUSION MATRIX: ", cm)
print("CLASSIFICATION REPORT: ", cr)
# df['Age'].hist()
# plt.show()

importance = pd.Series(model.feature_importances_, index=X.columns)
importance.sort_values().plot(kind='barh')
plt.title("What determined survival")
plt.show()
import pandas as pd
df=pd.read_csv(r"C:\Users\welcome\Desktop\twitter_training.csv")
print(df.head(20))

print(df.dropna(subset=['im getting on borderlands and i will murder you all ,'],inplace=True))

print(df.isnull().sum())

df=df.drop_duplicates()
print(df.duplicated().sum())

#spillting data
x=df[ 'im getting on borderlands and i will murder you all ,']
y=df['Positive']
#fitting in data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
positive_ok=le.fit_transform(df['Positive'])
#using tdf 
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
x=tfidf.fit_transform(df['im getting on borderlands and i will murder you all ,'])

#train test spilt
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#model selection
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
#pred
y_pred=model.predict(x_test)
#evaluation
from sklearn.metrics import accuracy_score, classification_report
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

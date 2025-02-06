# Install the missing package
!pip install pyforest

# Now import and use it
import pyforest

# if the CSV file is in the same directory as this notebook
data = pd.read_csv('/Users/deepak/Desktop/PROJECTS/Machine learning/TITANIC/titanic.csv') 
data

data.shape

data.isna().sum()

data.describe()

data.info()

data.dtypes

from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder();
data['Sex']=label_encoder.fit_transform(data['Sex'])
data['Sex'].value_counts()

data=data.drop(['Ticket','Cabin','Name'],axis=1)
data

data['Age'].median()

data['Age']=data['Age'].fillna(value=28)
data

data['Age'].isna().sum()

data['Embarked'].value_counts()

import seaborn as sns
import matplotlib.pyplot as plt
data['Embarked']=data['Embarked'].fillna(value='S')
data

data['Embarked'].isna().sum()
g=data.groupby('Survived')
g['Embarked'].value_counts()

from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
data['Embarked']=label_encoder.fit_transform(data['Embarked'])
data['Embarked'].value_counts()

sns.countplot(x='Embarked',hue='Survived',data=data)
plt.show()
data

from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder();
data['Sex']=label_encoder.fit_transform(data['Sex'])
data

data.corr()
data.plot(x='Survived',y=['SibSp','Parch'],kind='bar')
plt.show()

c=data.corr()
c['Survived'].sort_values(ascending=False)

sns.heatmap(data.corr())

data['family']=data['SibSp']+data['Parch']+1
data=data.drop(['SibSp','Parch','Embarked','PassengerId'],axis=1)
data

x=data.drop('Survived', axis=1).values
y=data['Survived'].values

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=100)

from sklearn.metrics import accuracy_score
lr=LogisticRegression()
lr. fit(x_train,y_train)#sending data to train 70% 
lrpred=lr.predict(x_test)

accuracy_score(y_test,lrpred)

from sklearn.model_selection import GridSearchCV
# Creating the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space}
# Instantiating the GridSearchCV object
logreg_cv = GridSearchCV(lr, param_grid, cv = 5)
logreg_cv.fit(x_train,y_train)
# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}". format(logreg_cv.best_params_))
print ("Best score is {}". format(logreg_cv.best_score_))
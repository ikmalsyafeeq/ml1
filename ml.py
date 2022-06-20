import pandas as pd
import numpy as np

pd.options.display.max_columns = None
pd.options.display.width=None

df=pd.read_csv('heart.csv')
pd.set_option('display.max_rows',20)
df

df.shape
df.head
df.select_dtypes(include=['number'])
df.select_dtypes(include=['object'])

df[df.age == df.age.max()]
df.info()

updated_df = df

updated_df['trtbps']=updated_df['trtbps'].fillna(updated_df['trtbps'].mean())
updated_df['chol']=updated_df['chol'].fillna(updated_df['chol'].mean())
updated_df['thalachh']=updated_df['thalachh'].fillna(updated_df['thalachh'].mean())
updated_df['oldpeak']=updated_df['oldpeak'].fillna(updated_df['oldpeak'].mean())
updated_df['slp']=updated_df['slp'].fillna(updated_df['slp'].mean())


updated_df.info()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5,metric = 'minkowski',p=2)
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)
print(y_pred)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

y_pred_new = classifier.predict(sc_x.transform(np.array([[40,1,21,145,322,1,1,170,0,3.0,0,0,1,0]])))
print(y_pred_new)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

x = updated_df.iloc[:,0:20]
y = updated_df.iloc[:,-1]

bestfeatures = SelectKBest (score_func=chi2,k=10)
fit = bestfeatures.fit(x,y)

dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns)

featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']

featureScores

print(featureScores.nlargest(10,'Score'))

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(x,y)

print(model.feature_importances_)

feat_importances = pd.Series(model.feature_importances_,index = x.columns)
feat_importances.nlargest(14).plot(kind='barh')

import seaborn as sns
corrmat = updated_df.corr()
top_corr_features = corrmat.index
plt.figure(figsize = (20,20))
g = sns.heatmap(updated_df[top_corr_features].corr(),annot = True,cmap="RdYlGn")




from sklearn.svm import SVC
classifier = SVC(kernel = 'linear',random_state=0)
classifier.fit(x_train,y_train)

y_predd = classifier.predict(x_test)
print(y_pred)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

y_predd_new = classifier.predict(sc_x.transform(np.array([[40,1,21,145,322,1,1,170,0,3.0,0,0,1,1]])))
print(y_pred_new)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

x = updated_df.iloc[:,0:20]
y = updated_df.iloc[:,-1]

bestfeatures = SelectKBest (score_func=chi2,k=10)
fit = bestfeatures.fit(x,y)

dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x.columns)

featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']

featureScores

print(featureScores.nlargest(10,'Score'))

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(x,y)

print(model.feature_importances_)

feat_importances = pd.Series(model.feature_importances_,index = x.columns)
feat_importances.nlargest(14).plot(kind='barh')

import seaborn as sns
corrmat = updated_df.corr()
top_corr_features = corrmat.index
plt.figure(figsize = (20,20))
g = sns.heatmap(updated_df[top_corr_features].corr(),annot = True,cmap="RdYlGn")
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

df=pd.read_excel('python.xlsx')
df

df.head()
df.describe()

plt.figure(figsize=(12,8)) 
X = df.drop(['Turnover'], axis=1) 
y = df['Turnover'] 
sns.distplot(y) 
plt.show()


df.Turnover.describe()


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df.columns

a,x = plt.subplots(figsize=(16, 7))
corr = df.corr()

matrix = sns.heatmap(corr, annot=True, square=True, fmt='.2f', 
                 linewidths= 0.1, vmax = 1, cmap = 'RdBu',
                  ax=x)
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
d_tree1 = DecisionTreeRegressor(max_depth = 3, random_state=42)
d_tree1.fit(X_train, y_train)

predictions = d_tree1.predict(X_test)
errors = abs(predictions - y_test)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'unit.')
mape = 100 * (errors / y_test)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 3), '%.')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline 


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix as cm

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
d_tree1 = DecisionTreeClassifier(max_depth = 2, random_state=42)
d_tree1.fit(X_train, y_train)

predictions = d_tree1.predict(X_test)


plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Accuracy Score: {0}'.format(score), size = 15)
plt.show()

corr

df[['GSYH','TUFE index','UFE index','Tuketici Guven Endeksi','Issizlik Orani']].corr()

sns.set(style="ticks", color_codes=True)
g = sns.pairplot(df[['GSYH','TUFE index','UFE index','Tuketici Guven Endeksi','Issizlik Orani']])

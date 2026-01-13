import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
import seaborn as sns
sns.set_palette('rainbow')

telecom=pd.read_csv('C:/Users/ELZAHBIA/AppData/Local/Temp/Rar$DRa7160.46026/datasets/churn.csv')

#first step rid all data which not make sense[area_code]
new_Telecom=telecom.drop('Area_Code',axis=1)

#step 2 transform all YES,NO to 0,1 in one column
print(telecom['Intl_Plan'].value_counts(dropna=False))#no:3010,yes:323=3333

new_Telecom['Intl_Plan']=pd.to_numeric(new_Telecom.iloc[:,7].map({'yes':1,'no':0}))
new_Telecom['Vmail_Plan']=pd.to_numeric(new_Telecom.iloc[:,8].map({'yes':1,'no':0}))
new_Telecom['Churn']=pd.to_numeric(new_Telecom.iloc[:,-1].map({'yes':1,'no':0}))

#international calls mins(intl_mins) with is this client has subscribe in package(intl_plan#predictions)
intl_mins_plan=new_Telecom.groupby('Intl_Mins')['Intl_Plan']
intl_charge=new_Telecom.groupby('Intl_Mins')['Intl_Charge']

#total bills and total calls 
new_Telecom['total_mins']=new_Telecom.iloc[:,2]+ new_Telecom.iloc[:,3]+new_Telecom.iloc[:,4]+new_Telecom.iloc[:,5]
new_Telecom['total_bills']=new_Telecom.iloc[:,10]+ new_Telecom.iloc[:,12]+new_Telecom.iloc[:,14]+new_Telecom.iloc[:,16]

#ABOUT
new_Telecom.info()
head=new_Telecom.head(10)
describe=new_Telecom.describe()
columns=new_Telecom.columns
tail=new_Telecom.tail(5)
null=new_Telecom.isnull().sum()
duplecatiom=new_Telecom.duplicated().sum()

#prediction step
feature=['total_mins','total_bills','CustServ_Calls','Intl_Plan','Vmail_Plan','Account_Length']
X=new_Telecom.loc[:,feature].values
y=new_Telecom.iloc[:,-3].values

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_stand=sc.fit_transform(X)
# Create the scatter plot
plt.figure(figsize=(8, 6))
# DBSCAN result plot: color by the cluster labels (db_labels)
scatter = plt.scatter(X_stand[:, 0], 
                      X_stand[:, 1],  # Second feature after scaling
                      cmap='coolwarm', # Color map
                      edgecolor='k',   # Edge color for points
                      alpha=0.7   )    # Transparency

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_stand,y,random_state=42,test_size=0.33)
#randomforest 94.8%
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=500,
                          criterion='entropy',
                          max_depth=10)
rfc.fit(X_train,y_train)
y_pred=rfc.predict(X_test)

#evaluation
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
cm=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)
ac=accuracy_score(y_test,y_pred)

# Plot the ideal prediction line (where prediction = actual)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Churn Prediction')
plt.show()

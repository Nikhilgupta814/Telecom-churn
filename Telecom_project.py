from joblib.logger import Logger
import numpy as np
from numpy.core.fromnumeric import prod, var
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import KBinsDiscretizer
from numpy import mean
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression



df = pd.read_csv(r'C:\Users\HP\OneDrive\Desktop\STATISTICS AND MACHINE LEARNING\Project\Telecom Churn (Project_2)\TelcoChurn.csv')
print(df.columns)
print(df.head())

df = df.drop(['customerID'], axis=1)
print(df.isnull().sum())
print(df.dtypes)

X = df.drop(['Churn'], axis=1)
y = df['Churn']

Num = X.select_dtypes(np.number)
Cat = X.select_dtypes(np.object)
y = y.map({'Yes':1, 'No':0})

Num['SeniorCitizen'].value_counts()
ind = Num['SeniorCitizen']                                             # Catagorical value(0,1)
Num = Num.drop(['SeniorCitizen'], axis=1) 
print(Num.describe())
print(Num.describe(percentiles=[0.01,0.05,0.10,0.25,0.50,0.75,0.85,0.9,0.99]))

def outlier_cap(x):
    x=np.clip(x,a_min=19.871000, a_max=8039.256000 )
    return(x)

Num = outlier_cap(Num)
print(Num.describe(percentiles=[0.01,0.05,0.10,0.25,0.50,0.75,0.85,0.9,0.99]))
print(var(Num))

discrete=KBinsDiscretizer(n_bins=10,encode='ordinal', strategy='quantile')
num_binned=pd.DataFrame(discrete.fit_transform(Num),index=Num.index, columns=Num.columns).add_suffix('_Rank')
print(num_binned.head())

X_bin_combined=pd.concat([y,num_binned],axis=1,join='inner')

# for col in (num_binned.columns):
#     plt.figure()
#     sns.barplot(x=col, y=y,data=X_bin_combined, estimator=mean )
# plt.show()

Final_Num = Num                                                            # Final selected numericla feature

# X_char_merged=pd.concat([y,Cat],axis=1,join='inner')

# for col in (Cat.columns):
#     plt.figure()
#     sns.barplot(x=col, y=y,data=X_char_merged, estimator=mean )
# plt.show()

Cat = Cat.drop(['gender','PhoneService', 'MultipleLines'], axis=1)

Cat = pd.get_dummies(Cat, drop_first=True)

selecotr = SelectKBest(chi2, k =20)
selecotr.fit_transform(Cat, y)
cols = selecotr.get_support(indices= True)
selected_feature_df_char = Cat.iloc[:,cols]

Final_Cat = selected_feature_df_char


# sns.barplot(x=ind, y=y)
# plt.show()

final_df = pd.concat([Final_Num,Final_Cat, ind, y], axis=1)
print(final_df.head())
x = final_df.drop(['Churn'], axis=1)
Y = final_df['Churn']

x_train, x_test, y_train, y_test = train_test_split(x, Y, test_size=0.3, random_state=80)

print('Ration of train data:', y_train.mean())
print('Ration of test data:', y_test.mean())

model_LR = LogisticRegression()
model_LR.fit(x_train, y_train)

coef_df_LR = pd.DataFrame(x.columns)
coef_df_LR.columns = ['Features']
coef_df_LR['Coef_feature'] = pd.Series(model_LR.coef_[0])
print(coef_df_LR)


############################################### Decision Tree Model ################################################
from sklearn.tree import DecisionTreeClassifier

model_DT= DecisionTreeClassifier(criterion='gini', random_state=99)

np.random.seed(99)
from sklearn.model_selection import GridSearchCV
param_dist= {'max_depth':[3,4,5,6], 'min_samples_split':[20, 50, 100, 150, 200, 250, 300, 400]}
tree_grid = GridSearchCV(model_DT, cv=10, param_grid=param_dist, n_jobs=-1)
tree_grid.fit(x_train, y_train)
print('Best Parameters using gridsearch : \n', tree_grid.best_params_)

model_DT=DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_split=20, random_state=99)
model_DT.fit(x_train, y_train)
print(final_df.shape)

# from sklearn import tree
# import pydotplus
# plt.figure(figsize=[50,10])
# tree.plot_tree(model_DT,filled=True, fontsize=15, rounded=True, feature_names=x.columns)
# plt.show()

####################################### Random Forest Model #######################################################

from sklearn.ensemble import RandomForestClassifier
model_RF=RandomForestClassifier(criterion='gini', random_state=99, max_depth=5, min_samples_split=20)
model_RF.fit(x_train, y_train)

import pandas as pd
feature_importance = pd.DataFrame(model_RF.feature_importances_, index=x_train.columns, columns=['importance']).sort_values('importance',ascending=False)
print(feature_importance)

####################### Model Evaluation #####################################################

y_predict_LR = model_LR.predict(x_test)
y_predict_DT = model_DT.predict(x_test)
y_predict_RF = model_RF.predict(x_test)

from sklearn import metrics

##################### Model Evaluation Logistic Regression##################

print('Accuracy of LR: ', metrics.accuracy_score(y_test, y_predict_LR))
print('Precision of LR: ', metrics.precision_score(y_test, y_predict_LR))
print('Recall of LR: ', metrics.recall_score(y_test, y_predict_LR))
print('f1_score of LR: ', metrics.f1_score(y_test, y_predict_LR))


# ##################### Model Evaluation Logistic Regression##################
# print('\n')
# print('Accuracy of DT: ', metrics.accuracy_score(y_test, y_predict_DT))
# print('Precision of DT: ', metrics.precision_score(y_test, y_predict_DT))
# print('Recall of DT: ', metrics.recall_score(y_test, y_predict_DT))
# print('f1_score of DT: ', metrics.f1_score(y_test, y_predict_DT))

# ##################### Model Evaluation Logistic Regression##################
# print('\n')
# print('Accuracy of RF: ', metrics.accuracy_score(y_test, y_predict_RF))
# print('Precision of RF: ', metrics.precision_score(y_test, y_predict_RF))
# print('Recall of RF: ', metrics.recall_score(y_test, y_predict_RF))
# print('f1_score of RF: ', metrics.f1_score(y_test, y_predict_RF))

metrics.plot_confusion_matrix(model_LR,x_test,y_test)
plt.title('Regression Model')
plt.show()

# metrics.plot_confusion_matrix(model_DT,x_test,y_test)
# plt.title('DT Model')

# metrics.plot_confusion_matrix(model_RF,x_test,y_test)
# plt.title('RF Model')

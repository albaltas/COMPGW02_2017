#%%

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

import random


#%%
df_train = pd.read_csv("data/train_ohe.csv")
df_val = pd.read_csv("data/validation_ohe.csv")
df_test = pd.read_csv("data/test_ohe.csv")

print (df_train.click.values.sum()*100/len(df_train), "%")
print (df_val.click.values.sum()*100/len(df_val), "%")

#%%
features = list(df_train.columns)
features_remove = ['click', 'bidid', 'logtype', 'userid', 'urlid', 'bidprice', 'payprice', 'usertag']
features = [x for x in features if x not in features_remove]
classification_column = 'click'

rand_seed = 27
random.seed(rand_seed)
np.random.seed(rand_seed)
num_to_sample = 15000



X_train_full = df_train[features].copy()
X_val_full = df_val[features].copy()
X_test = df_test[features].copy()

y_val_full = df_val.click

df_train_sample = df_train.sample(num_to_sample, random_state=rand_seed).copy()

X_train_inliers = df_train_sample[features][df_train.click == 0].copy()
y_train_inliers = df_train_sample[classification_column][df_train.click == 0].copy()

X_train_outliers = df_train[features][df_train.click == 1].copy()
y_train_outliers = df_train[classification_column][df_train.click == 1].copy()

X_train = X_train_inliers.copy()
y_train = y_train_inliers.copy()



X_val_inliers = df_val[features][df_val.click == 0].copy()
y_val_inliers = df_val[classification_column][df_val.click == 0].copy()

X_val_outliers = df_val[features][df_val.click == 1].copy()
y_val_outliers = df_val[classification_column][df_val.click == 1].copy()

features_to_process = X_test.columns.values
features_to_process = [x for x in features_to_process if "usertag_" not in x]

#%%
for col in features_to_process:
    
    data = X_train_full[col].append(X_val_full[col]).append(X_test[col])
    
    if X_test[col].dtypes == 'object':
        median_val = data.value_counts().index[0]
    else:
        median_val = data.median()
        
    
    X_train_full[col].fillna(median_val, inplace=True)
    X_train[col].fillna(median_val, inplace=True)
    X_train_inliers[col].fillna(median_val, inplace=True)
    X_train_outliers[col].fillna(median_val, inplace=True)

    X_val_full[col].fillna(median_val, inplace=True)
    X_val_inliers[col].fillna(median_val, inplace=True)
    X_val_outliers[col].fillna(median_val, inplace=True)

    X_test[col].fillna(median_val, inplace=True)
    
le = {}
for col in features_to_process:
    
    if X_test[col].dtypes == 'object':
    
        data = X_train_full[col].append(X_val_full[col]).append(X_test[col])
        
        le[col] = preprocessing.LabelEncoder() #define and store a label encoder for every column so we can inverse_transform
        le[col].fit(data.values)
        
        X_train_full[col] = le[col].transform(X_train_full[col])
        X_train[col] = le[col].transform(X_train[col])
        X_train_inliers[col] = le[col].transform(X_train_inliers[col])
        X_train_outliers[col] = le[col].transform(X_train_outliers[col])
        
        X_val_full[col] = le[col].transform(X_val_full[col])
        X_val_inliers[col] = le[col].transform(X_val_inliers[col])
        X_val_outliers[col] = le[col].transform(X_val_outliers[col])
        
        X_test[col] = le[col].transform(X_test[col])
        

ss = {}
do_scale = True
        
if do_scale:
    data_scale = X_train_full[0:1].append(X_test[0:1]).append(X_val_full[0:1])
    features_to_process = [x for x in data_scale.columns.values if "usertag_" not in x]
    for col in features_to_process:

        data = X_train_full[col].append(X_val_full[col]).append(X_test[col])

        ss[col] = preprocessing.MinMaxScaler() #define and store a label encoder for every column so we can inverse_transform
        ss[col].fit(data.values)
        
        if col in X_train.columns.values:

            X_train[col] = ss[col].transform(X_train[col])
            X_train_inliers[col] = ss[col].transform(X_train_inliers[col])
            X_train_outliers[col] = ss[col].transform(X_train_outliers[col])
            
            X_val_full[col] = ss[col].transform(X_val_full[col])
            X_val_inliers[col] = ss[col].transform(X_val_inliers[col])
            X_val_outliers[col] = ss[col].transform(X_val_outliers[col])
            
        if col in X_test.columns.values:
            X_test[col] = ss[col].transform(X_test[col])

#%%

t_feat = pd.concat([X_train_inliers, X_train_outliers])
t_lab = pd.concat([y_train_inliers, y_train_outliers])



from sklearn.neural_network import MLPClassifier

#clf = MLPClassifier(hidden_layer_sizes=(60, 10, 4))


clf = MLPClassifier(hidden_layer_sizes=(60,10,4))
clf.fit(t_feat, t_lab)

val_preds = clf.predict(X_val_full)
print(confusion_matrix(y_val_full, val_preds))



 #%%
good_clf=clf
#%%
val_preds = good_clf.predict(X_val_full)
print(confusion_matrix(y_val_full, val_preds))

val_probas = good_clf.predict_proba(X_val_full)
val_click_prob = [x[1] for x in val_probas]
#%%
#20,10,4
#[[295894   3629]
# [   125    101]]
#pd.DataFrame({"bidid":df_val.bidid,"clickprob":val_click_prob, "clickpred":val_preds}).to_csv('nn_val_preds.csv')

#100,20,10
#[[285291  14232]
# [   117    109]]
#pd.DataFrame({"bidid":df_val.bidid,"clickprob":val_click_prob, "clickpred":val_preds}).to_csv('nn_val_preds_v2.csv')

#60,10,4
#[[292516   7007]
# [   115    111]]
pd.DataFrame({"bidid":df_val.bidid,"clickprob":val_click_prob, "clickpred":val_preds}).to_csv('nn_val_preds_v3.csv')
#%%
test_preds = good_clf.predict(X_test)
#print(confusion_matrix(y_val_full, val_preds))

test_probas = good_clf.predict_proba(X_test)
test_click_prob = [x[1] for x in test_probas]
pd.DataFrame({"bidid":df_test.bidid,"clickprob":test_click_prob, "clickpred":test_preds}).to_csv('nn_test_preds.csv')


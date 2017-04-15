#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

# coding: utf-8

# In[1]:

import pandas as pd
import sys
import pickle
import csv
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
#import matplotlib.pyplot as plt
#from IPython.display import display # Allows the use of display() for DataFrames
import numpy as np

from sklearn import cross_validation
from sklearn.model_selection import StratifiedShuffleSplit,StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import make_scorer
from numpy import mean
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score ,fbeta_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score,precision_score,recall_score,precision_recall_fscore_support,roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
sys.path.append("../tools/")


# In[2]:

data_dict = pickle.load(open("final_project_dataset.pkl", "r") )


# In[3]:

data_df = pd.DataFrame(data_dict).transpose()


# In[4]:

#isplay(data_df)


# Since we do not need the email-address field for classification, we can drop it, as it is unique for each individual user (and in some cases contain NaN)

# In[5]:

del data_df['email_address']


# In[6]:

print "Count of NaNs in each column"
data_df.isnull().sum()


# We can see clearly that there exists NaN values, but they are considered as String 'NaN' instead of NaN. So we replace all NaN values with numpy's NaN

# In[7]:

data_df=data_df.replace('NaN',np.nan)


# In[8]:

print "Count of NaNs in each column after cleaning"
data_df.isnull().sum()


# In[9]:

print "Percentage of NaNs in each column"
(data_df.isnull().sum()*1.0)/(len(data_df))*100


# Lets seperate the count of NaNs in each column based on the POI label

# In[10]:

poi_df = pd.DataFrame(data_df[data_df.poi==1])
notpoi_df = pd.DataFrame(data_df[data_df.poi==0])
del poi_df['poi']
del notpoi_df['poi']


# In[11]:

notpoi_per = notpoi_df.isnull().sum()/128*100
poi_per= poi_df.isnull().sum()/18*100
nan_comp_df = pd.concat([notpoi_per,poi_per],axis=1)
nan_comp_df.columns = ['Not POI %','POI %']
#display(nan_comp_df)


# It might be the case that if we continue with the above data then the classifier might simply classify the POI as 1 or 0 simply on the basis of NaN . So for now , I am removing those columns which contain more 70% NaN values .

# In[12]:

to_delete = nan_comp_df[(nan_comp_df>70.0).any(axis=1)].index.tolist()
print "Columns to be deleted are "
print to_delete


# In[13]:

data_df.drop(to_delete, axis=1, inplace=True)


# In[14]:

initial_nan = data_df.isnull().sum()
print "Current count of NaNs after removing above mentioned columns"
print initial_nan


# Next we find rows with NaN in more than 75% rows

# In[15]:

current_features = data_df.columns
total_features = len(current_features)*1.0
nan_rows = np.asarray(data_df.isnull().sum(axis=1).tolist())/total_features*100


# In[16]:

drop_rows = nan_rows>70
print "No. of rows with more than 70% NaN values (i.e. have more than 10 columns empty) are ",drop_rows.sum()


# In[17]:

#Before removing the row we check if any of these rows have POI = 1 , i.e. they actually are Persons of Interest
data_df[drop_rows][data_df['poi']==1]


# Since None of these rows are POI, we can delete them as these will not further lead to any class imbalance

# In[18]:

data_df=data_df.drop(data_df.index[drop_rows.nonzero()])


# In[19]:

# display(data_df)


# After remove the rows and columns we can se below that the overnall count of NaN's has decreased conisdderably for each column . The percentage of NaN values reduced for each columns are as follows : 

# In[20]:

print "Reduction in percentage of NaNs in each column after cleaning "
100-(data_df.isnull().sum()*1.0)/initial_nan*100


# Outlier Dectection  
# I tried various combinations of columns to identify the outliers, but I found the outlier when I plotted a scatter plot between 'bonus' and 'salary' . The outlier had the name as 'Total' suggesting that it might be a record containing the sum of all other values . I deleted that row to further imporve the quality the dataset .

# In[21]:

# def plot_scatter(col1,col2):
#     col1_df = data_df[col1]
#     col2_df = data_df[col2]
#     plt.scatter(col1_df,col2_df)
#     plt.xlabel(col1)
#     plt.ylabel(col2)
#     plt.xticks(rotation='vertical')
#     title = 'Scatter plot for '+col1+' vs. '+col2
#     plt.title(title)
#     plt.scatter(data_df[col1], data_df[col2])
#     #plt.legend(loc='lower right')
#     plt.show() 


# # In[22]:

# plot_scatter('bonus','salary')


# In[23]:

data_df.salary.argmax()


# In[24]:

data_df.drop('TOTAL',inplace=True)


# In[25]:

# plot_scatter('bonus','salary')


# The points are valid and should not be removed .

# ### Imputing Missing Values.  
# Now next thing we do is impute the missing values . First we divide the features into three sets :   
# 1. Financial Features
# 2. Email features
# 3. Target Label

# In[26]:

all_features = list(data_df.columns.values)
email_features = ['from_messages','from_poi_to_this_person','from_this_person_to_poi','shared_receipt_with_poi','to_messages']
target =['poi']
financial_features = [item for item in all_features if item not in email_features and item not in target ]
print financial_features


# For Financial features, we will be replacing the missing values with the mean .

# In[27]:

financial_df = data_df[financial_features].copy()
financial_df.fillna(financial_df.mean(),inplace=True)


# For email_features, we will be replacing the missing values with mean .

# In[28]:

email_df = data_df[email_features].copy()
email_df.fillna(email_df.mean(),inplace=True)


#     Next , we combine the final imputed dataset .

# In[29]:

enron_data_df = pd.concat([financial_df,email_df],axis=1, join_axes=[email_df.index])


# In[30]:

enron_data_df


# ###Feature Engineering
# 1. Email Features

# In[31]:

enron_data_df['fraction_from_poi'] = enron_data_df.from_poi_to_this_person / enron_data_df.from_messages
enron_data_df['fraction_to_poi'] = enron_data_df.from_this_person_to_poi / enron_data_df.to_messages
enron_data_df['related_to_poi']= (enron_data_df.from_poi_to_this_person+enron_data_df.from_this_person_to_poi+enron_data_df.shared_receipt_with_poi)/(enron_data_df.to_messages+enron_data_df.from_messages)


# In[32]:

enron_data_df


# 2. Financial Features

# In[33]:

enron_data_df['Effective Salary']= enron_data_df['bonus'] +                                 enron_data_df['salary'] +                                 enron_data_df['long_term_incentive'] -                                 enron_data_df['expenses']


# In[34]:

del enron_data_df['bonus']
del enron_data_df['salary']


# Peform Feature selection

# In[35]:

skb = SelectKBest(f_classif,k='all').fit(enron_data_df,data_df['poi'])
scores = skb.scores_
all_features = enron_data_df.columns.values
sort_index = np.argsort(scores)[::-1]
rank = 1
ranked_features = []
print "Ranking of features is as follows "
for x in sort_index:
    print rank,". Score for ",all_features[x]," is ",scores[x]
    ranked_features.append(all_features[x])
    rank += 1
print all_features


# Now I will make three sets of features , each containing 5 ,7 and 10 top-most features respectively .

# In[36]:

features_5 = ranked_features[:5]
features_7 = ranked_features[:7]
features_10 = ranked_features[:10]
print "Features in first set are ",features_5
print "\nFeatures in second set are ",features_7
print "\nFeatures in third set are ",features_10


# In[37]:

#Ref : http://stackoverflow.com/questions/30523735/python-dictionary-as-html-table-in-ipython-notebook
class DictTable(dict):
    # Overridden dict class which takes a dict in the form {'a': 2, 'b': 3},
    # and renders an HTML Table in IPython Notebook.
    def _repr_html_(self):
        html = ["<table width=100%>"]
        html.append("<tr>")
        html.append("<th>Classifier</th>")
        html.append("<th>Accuracy</th>")
        html.append("<th>Precision</th>")
        html.append("<th>Recall</th>")
        html.append("</tr>")
        for key, value in self.iteritems():
            html.append("<tr>")
            html.append("<td>{0}</td>".format(key))
            for key_inner,value_inner in value.iteritems():
                html.append("<td>{0}</td>".format(value_inner))
                
           
            html.append("</tr>")
        html.append("</table>")
        return ''.join(html)


# In[38]:

labels = data_df['poi']
from sklearn.cluster import KMeans
classifier_dict = {'Logistic Regrssion':LogisticRegression(random_state = 0,max_iter=500,class_weight='balanced'),
                    'SVM_rbf':SVC(class_weight='balanced',kernel='rbf'),
                   'SVM_sigmoid':SVC(class_weight='balanced',kernel='sigmoid'),
                  'Gaussian Naive Bayes':GaussianNB(),
                  'SVM_linear':LinearSVC(random_state=0,class_weight='balanced'),
                  'Decision Tree':DecisionTreeClassifier(class_weight='balanced',min_samples_split=5),
                  'Random Forest': RandomForestClassifier(n_estimators=100,class_weight='balanced'),
                  'KNN':KNeighborsClassifier()
                  }


# In[39]:

def get_score_table(features_set):
    features = enron_data_df[features_set]
    classifier_comp = {}
    for x in classifier_dict.keys():
        pipeline =  Pipeline(steps=[('scaler', StandardScaler()),
                                         ("classifier",classifier_dict[x])])
        cv = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
        acc = []
        pre = []
        rec = []
        for train_index, test_index in cv.split(features, labels):
            X_train, X_test = features.iloc[train_index], features.iloc[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
        #cv=StratifiedKFold(n_splits=n_splits).split(enron_data_df[features_5], data_df['poi'])
            pipeline.fit(X_train,y_train)
            predictions = pipeline.predict(X_test)
            acc.append(accuracy_score(predictions,y_test))
            pre.append(precision_score(predictions,y_test))
            rec.append(recall_score(predictions,y_test))
        #print pipeline.named_steps
        classifier_comp[x]={}
        classifier_comp[x]['accuracy']=np.mean(acc)
        classifier_comp[x]['precision']=np.mean(pre)
        classifier_comp[x]['recall']=np.mean(rec)
    return classifier_comp


# In[40]:

#Using top 5 features :d
print "Using Top 5 fetaures"
#DictTable(get_score_table(features_5))
print features_5

# In[41]:

#Using top 7 features :d
print "Using Top 7 fetaures"
# DictTable(get_score_table(features_7))
print features_7

# In[42]:

#Using top 10 features :d
print "Using Top 10 fetaures"
# DictTable(get_score_table(features_10))
print features_10


# It can be seen clearly that Top 10 features set is giving the worst performance among all 3, so we will discard it .  
# Now we are left with the Top 5 and Top 7 features set respectively . Now after examining the above three tables , I observed the following facts :
# 1. KNN classifier is performing the most poorly among all classifier for all sets .
# 2. Decision Tree's performance remained more or less stable.
# 3. SVM with rbf reported the best results with Top 5 features.
# 4. Since Gaussian Naive Bayes can't be used for parameter tuning, I will discard it since it's scores are not good enough as well.
# 5. SVM with linear kernel reported the best results with Top 5 features.
# 6. Logistic regression reported more or less constant results.
# 7. SVM with linear kernel reported the best results with Top 7 features.
# 8. Random Forest reports good results for both top 5 and top 10 features .
# 
# After considering the above mentioned facts, I selected Top 5 features set as the final features set for classification .
# I selected the following algorithm for fine tuning as well :
# 1. SVM with 'sigmoid' kernel
# 2. Logistic Regression
# 3. Decision Trees
# 

# In[145]:

features = enron_data_df[features_7]


# Now based on the performance of above classifiers, I have decided to select the following classifiers for fine tuning :
# 1. SVM with 'rbf' kernel
# 2. Logistic Regression
# 3. Decision Trees

# In[146]:

def get_best_estimator(my_clf,param_grid,n_splits = 10):
    skb = SelectKBest(f_classif)
    clf = pipeline =  Pipeline(steps=[('scaler', StandardScaler()),
                                     ("classifier",my_clf)])
    cv=StratifiedKFold(n_splits=n_splits).split(features,labels)
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=10,scoring=make_scorer(f1_score,average='weighted'),cv=list(cv))
    grid_search.fit(features.values,labels.values)
    return grid_search


# In[147]:

my_dataset = pd.concat([labels,features],axis=1)
my_feature_list = my_dataset.columns.values
my_dataset = my_dataset.transpose().to_dict()
grid_comp = {}


# In[135]:

my_clf = LogisticRegression(random_state = 0,max_iter=500,class_weight='balanced')
param_grid = dict(
                classifier__tol=[1.0,0.1,0.01,0.001],
                  classifier__C = np.power([10.0]*5,list(xrange(-3,2))).tolist(),
                  classifier__solver =['newton-cg', 'lbfgs', 'liblinear', 'sag'],

                 )
grid_logistic = get_best_estimator(my_clf,param_grid)


# In[136]:

grid_comp['Logistic Regression']=grid_logistic.best_score_


# In[137]:

my_clf = SVC(class_weight='balanced',random_state=0,kernel='rbf',max_iter=500)
param_grid = dict(
                  classifier__tol=[1.0,0.1,0.01,0.001],
                  classifier__C = np.power([10.0]*4,list(xrange(-2,2))).tolist(),
                    classifier__gamma = 1.0/np.asarray(list(xrange(1,6))),
                 )
grid_svm = get_best_estimator(my_clf,param_grid)


# In[138]:

grid_comp['SVM with sigmoid kernel']=grid_svm.best_score_
print grid_svm.best_estimator_
print grid_comp


# In[139]:

my_clf = DecisionTreeClassifier(random_state=0,class_weight='balanced')
param_grid = dict(
                classifier__min_samples_split=list(xrange(2,20,5)),
                classifier__max_leaf_nodes =list(xrange(50,100,20)),
                classifier__max_depth = list(xrange(1,10,2))
                )
grid_dt = get_best_estimator(my_clf,param_grid)


# In[140]:

grid_comp['Decision Tree']=grid_dt.best_score_


# In[141]:

#Ref : http://stackoverflow.com/questions/30523735/python-dictionary-as-html-table-in-ipython-notebook
class DictTable_grid(dict):
    # Overridden dict class which takes a dict in the form {'a': 2, 'b': 3},
    # and renders an HTML Table in IPython Notebook.
    def _repr_html_(self):
        html = ["<table width=100%>"]
        html.append("<tr>")
        html.append("<th>Classifier</th>")
        html.append("<th>ROC AUC score</th>")
        html.append("</tr>")
        for key, value in self.iteritems():
            html.append("<tr>")
            html.append("<td>{0}</td>".format(key))
            html.append("<td>{0}</td>".format(value))
            html.append("</tr>")
        html.append("</table>")
        return ''.join(html)


# In[142]:

# DictTable_grid(grid_comp)
print grid_comp

# Based on the above table, the best classifier is Logistic Kernel.

# In[144]:

best_clf = grid_logistic.best_estimator_
print "Best Classifier is"
print best_clf
pickle.dump(best_clf, open("my_classifier.pkl", "w"))
pickle.dump(my_dataset, open("my_dataset.pkl", "w"))
pickle.dump(my_feature_list, open("my_feature_list.pkl", "w"))
print "Data Saved Successfuly !!"

#!/usr/bin/env python
# coding: utf-8

# In[64]:


#Import libraries needed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [12, 8]
import seaborn as sns
sns.set()
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import shap
from sklearn.model_selection import KFold


# In[2]:


dataset = pd.read_csv("Data/Dataset_spine.csv")
#Removing the unnecesary last column
del dataset["Unnamed: 13"]


#CSV file describes the attributes as such

#We need to rename the columns so they match
dataset.rename(columns = {
    "Col1" : "pelvic_incidence", 
    "Col2" : "pelvic_tilt",
    "Col3" : "lumbar_lordosis_angle",
    "Col4" : "sacral_slope", 
    "Col5" : "pelvic_radius",
    "Col6" : "degree_spondylolisthesis", 
    "Col7" : "pelvic_slope",
    "Col8" : "direct_tilt",
    "Col9" : "thoracic_slope", 
    "Col10" :"cervical_tilt", 
    "Col11" : "sacrum_angle",
    "Col12" : "scoliosis_slope", 
    "Class_att" : "class"}, inplace=True)

dataset.head()


# In[3]:


#Check for NULL count and data types
dataset.info()


# In[4]:


#Check for Balance
dataset["class"].value_counts().sort_index().plot.bar()


# In[5]:


#Check for correlation
sns.heatmap(dataset.corr(), annot = True, center=0, cmap = 'vlag')
#According to https://josr-online.biomedcentral.com/articles/10.1186/s13018-018-0762-9, 
#pelvic incidence = pelvic tilt + sacral slope. Which means we could potentially remove this feature.


# In[6]:


#Box plot for outliers
dataset.boxplot(patch_artist = True, sym = 'kx') 
plt.xticks(rotation=90)


# In[7]:


#Removing one outlier in degree_spondylolisthesis
ind_outlier = dataset['degree_spondylolisthesis'].idxmax(axis=0)
med = dataset['degree_spondylolisthesis'].median()
dataset['degree_spondylolisthesis'][ind_outlier]= med


# In[8]:


#Box plot for outliers
dataset.boxplot(patch_artist = True, sym = 'kx') 
plt.xticks(rotation=90)


# In[9]:


X = dataset.iloc[:, :-1]
Y = pd.get_dummies(dataset, columns = ['class'])
Y = Y.iloc[:,-2:]


# In[37]:


def create_MLP():
    model = Sequential()
    model.add(Dense(8, input_shape = (12,), activation = 'relu', name ='FC1'))
    model.add(Dense(4, name = 'FC2'))
    model.add(Dense(2, activation = "softmax", name = 'Output'))
    
    optimizer = Adam(learning_rate = 0.001)
    model.compile(optimizer, loss = 'binary_crossentropy', metrics = 'accuracy')
    return model


# In[39]:


#Getting the hold out set(test set)
checkpoint_filepath = 'tmp/checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

my_callbacks = [model_checkpoint_callback]


# In[72]:


#Defining K-folds Cross Validator
kfold = KFold(n_splits = 10, shuffle = True)
n_fold = 1
reports = []
c_matrices = []
for train, test in kfold.split(X):
    
    x_train = X.iloc[train]
    y_train = Y.iloc[train]
    x_test = X.iloc[test]
    y_test = Y.iloc[test]
    
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)


    MLP_model = create_MLP()
    
    print('\nTraining for fold ', n_fold)
    history_MLP = MLP_model.fit(x_train_scaled, 
                                y_train,
                                verbose = 0, 
                                batch_size = 4, 
                                epochs = 200, 
                                validation_split = 0.1, 
                                callbacks = my_callbacks)
    
    y_pred = MLP_model.predict(x_test_scaled)
    y_tens = tf.convert_to_tensor(y_test)
    target_names = ['Abnormal', 'Normal']
    report = classification_report(np.argmax(y_tens,axis=1), np.argmax(y_pred,axis=1), target_names= target_names, output_dict=True)
    c_matrix = confusion_matrix(np.argmax(y_tens,axis=1), np.argmax(y_pred,axis=1))
    
    reports.append(report)
    c_matrices.append(c_matrix)
    n_fold = n_fold + 1


# In[93]:


MLP_model.summary()


# In[94]:


ab_precision = []
n_precision = []
ab_recall = []
n_recall = []
ab_f1_score = []
n_f1_score = []
ab_support = []
n_support = []

for i in range(10):
    ab_precision.append(reports[i]["Abnormal"]["precision"])
    n_precision.append(reports[i]["Normal"]["precision"])
    ab_recall.append(reports[i]["Abnormal"]["recall"])
    n_recall.append(reports[i]["Normal"]["recall"])
    ab_f1_score.append(reports[i]["Abnormal"]["f1-score"])
    n_f1_score.append(reports[i]["Normal"]["f1-score"])
    ab_support.append(reports[i]["Abnormal"]["support"])
    n_support.append(reports[i]["Normal"]["support"])

print("Precision")
print(stats.describe(ab_precision))
print(stats.describe(n_precision))

print("\nRecall")
print(stats.describe(ab_recall))
print(stats.describe(n_recall))

print('\nF1 score')
print(stats.describe(ab_f1_score))
print(stats.describe(n_f1_score))

print('\nSupport')
print(stats.describe(ab_support))
print(stats.describe(n_support))



data = [ab_precision, n_precision, ab_recall, n_recall, ab_f1_score, n_f1_score]
plt.boxplot(data, patch_artist=True)
plt.xticks([1,2,3,4,5,6],['ab_precision', 'n_precision', 'ab_recall', 'n_recall', 'ab_f1_score', 'n_f1_score'])


# In[84]:


shap_values = MLP_explainer.shap_values(x_test_scaled)
target_names = ['Abnormal', 'Normal']
shap.summary_plot(shap_values, x_test,class_names = target_names)


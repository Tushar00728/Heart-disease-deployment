#!/usr/bin/env python
# coding: utf-8

# ## Predicting heart disease using machine learning
# 
# This notebook looks into using various python based machine learning and data science libraries in an attempt to build a ML model capable of predicting whether someone has heart-disease based on their medical attributes
# 
# We are going to take the following approach
# 
# 1. Problem definition 
# 2. Data 
# 3. Evaluation 
# 4. Features
# 5. Modelling 
# 6. Experimentation 
# 
# ## Problem Definition 
# 
# In a statement, 
# > Given clinical parameters about a patient can we predict whether or not they have heart disease?
# 
# ## Data
# The original data came from the Cleavland data from the UCI Machine Learning Repository. https://archive.ics.uci.edu/ml/datasets/heart+Disease
# 
# There is also a version of it available on Kaggle. https://www.kaggle.com/ronitf/heart-disease-uci
# 
# ## Evaluation 
# 
# > If we can reach 95% accuracy at predicting whether or not a patient has heart disease during the proof of concept, we'll pursue the project
# 
# ## Features
# 
# This is where you'll get different information about each of the features in your data. You can do this via doing your own research (such as looking at the links above) or by talking to a subject matter expert (someone who knows about the dataset).
# 
# **Create data dictionary**
# 
# 1. age - age in years
# 2. sex - (1 = male; 0 = female)
# 3. cp - chest pain type
#  0: Typical angina: chest pain related decrease blood supply to the heart
#  1: Atypical angina: chest pain not related to heart
#  2: Non-anginal pain: typically esophageal spasms (non heart related)
#  3: Asymptomatic: chest pain not showing signs of disease
# 4. trestbps - resting blood pressure (in mm Hg on admission to the hospital) anything above 130-140 is typically cause for concern
# 5. chol - serum cholestoral in mg/dl
#  serum = LDL + HDL + .2 * triglycerides
#  above 200 is cause for concern
# 6. fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# '>126' mg/dL signals diabetes
# 7. restecg - resting electrocardiographic results
#  0: Nothing to note
#  1: ST-T Wave abnormality
#  can range from mild symptoms to severe problems
#  signals non-normal heart beat
#  2: Possible or definite left ventricular hypertrophy
#  Enlarged heart's main pumping chamber
# 8. thalach - maximum heart rate achieved
# 9. exang - exercise induced angina (1 = yes; 0 = no)
# 10. oldpeak - ST depression induced by exercise relative to rest looks at stress of heart during excercise unhealthy heart will   stress more
# 11. slope - the slope of the peak exercise ST segment
#  0: Upsloping: better heart rate with excercise (uncommon)
#  1: Flatsloping: minimal change (typical healthy heart)
#  2: Downslopins: signs of unhealthy heart
# 12. ca - number of major vessels (0-3) colored by flourosopy
#  colored vessel means the doctor can see the blood passing through
#  the more blood movement the better (no clots)
# 13. thal - thalium stress result 
#  1,3: normal
#  6: fixed defect: used to be defect but ok now
#  7: reversable defect: no proper blood movement when excercising
# 14. target - have disease or not (1=yes, 0=no) (= the predicted attribute)

# ## Preparing the tools
# 
# We're going to use numpy,pandas,matplotlib for for data analysis and manipulation 

# In[67]:


#Import all the tools we need 


#Regular EDA(exploratory data analysis) and plotting libraries 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle 

#We want our plots to appear inside the notebooks using matplotlib inline 
get_ipython().run_line_magic('matplotlib', 'inline')

#Models from sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier


#Model evaluations 

from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report 
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import plot_roc_curve


# ## Load Data
# 

# In[2]:


df= pd.read_csv("heart-disease.csv")


# In[3]:


df


# In[4]:


df.shape #(rows,columns)


# ## Data exploration exploratory data analysis or EDA
# 
# The goal here is to find more about the data and become a subject matter expert on the dataset you are working with 
# 1. What questions are you trying to solve?
# 
# 2. What kind of data we have and how do we treat different types? 
# 3. What's missing from the data and how do you deal with it ?
# 4. Where are the outliers and shy should you care about them ? 
# 5. How can you add, change or remove features to get more out of your data ?
# 
# 

# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df["target"].value_counts() #How many of each class are there 


# In[7]:


df["target"].value_counts().plot(kind= "bar" , color=["salmon","lightblue"]); 


# In[8]:


df.info()


# In[9]:


df.isna() #Are there any missing values  


# In[10]:


df.describe()


# ### Heart disease frequency according to Sex 

# In[11]:


df.sex.value_counts()


# In[12]:


#Compare target column with sex column 

pd.crosstab(df.target , df.sex)


# In[13]:


#Create a plot of crosstab 

pd.crosstab(df.target, df.sex).plot(kind="bar" , figsize=(10,6), color= ["salmon","lightblue"]); 
plt.title("Heart disease frequency for Sex")
plt.xlabel("0=No disease, 1=Disease")
plt.ylabel("Amount")
plt.legend(["Female","Male"])


# In[14]:


df.head()


# ### Age v/s Max.Heart Rate for Heart Disease

# In[15]:


plt.figure(figsize=(10,6)) #Create another figure 

#scatter with positive examples

plt.scatter(df.age[df.target==1], df.thalach[df.target==1], color= "salmon"); #age columns where target=1

#Scatter with negative examples

plt.scatter(df.age[df.target==0], df.thalach[df.target==0], color= "lightblue");

#Add some helpful info

plt.title("Heart Disease in function of Age and Max Heart Rate")
plt.xlabel("Age")
plt.ylabel("Max Heart Rate")
plt.legend(["Disease", "No Disease"]);


# In[16]:


#Check the distribtion of age with a Histogram (check normal distribution)
df.age.plot.hist(); #histogram good for spreading data and checking outliers 


# ### Heart Disease Frequency per chest pain type 
# 
# 3. cp - chest pain type 0: Typical angina: chest pain related decrease blood supply to the heart 
# * 1: Atypical angina: chest pain not related to heart
# * 2: Non-anginal pain: typically esophageal spasms (non heart related) 
# * 3: Asymptomatic: chest pain not showing signs of disease

# In[17]:


pd.crosstab(df.cp,df.target)


# In[18]:


#Make crosstab more visual 

pd.crosstab(df.cp,df.target).plot(kind="bar", figsize=(10,6), color=["salmon", "lightblue"])

plt.title("Heart disease Frequency per chest pain type")
plt.xlabel("Chest Pain Type")
plt.ylabel("Amount")
plt.legend(["No disease","Disease"])
plt.xticks(rotation= 0);


# In[19]:


#Make a correlation matrix

df.corr()


# In[20]:


#Making correlation matrix prettier

corr_matrix = df.corr()
fig,ax= plt.subplots(figsize=(15,10))
ax= sns.heatmap(corr_matrix,annot=True,linewidth=0.5,fmt=".2f",cmap="YlGnBu");


# ## Modelling  

# In[21]:


df.head()


# In[22]:


X = df.drop("target" , axis =1)
y = df["target"]


# In[23]:


X


# In[24]:


y


# In[25]:


np.random.seed(42)

X_train , X_test , y_train , y_test = train_test_split(X,y ,test_size=0.2)


# In[26]:


X_train


# We are going to try 3 different ML models :
# 
# 1. Logistic Regression 
# 2. K- Nearest Neighbours Classifier
# 3. Random Forest Classifier

# In[27]:


models = {"KNN": KNeighborsClassifier(),
          "Logistic Regression": LogisticRegression(), 
          "Random Forest": RandomForestClassifier()}

# Create function to fit and score models
def fit_and_score(models, X_train, X_test, y_train, y_test):
    """
    Fits and evaluates given machine learning models.
    models : a dict of different Scikit-Learn machine learning models
    X_train : training data
    X_test : testing data
    y_train : labels assosciated with training data
    y_test : labels assosciated with test data
    """
    # Random seed for reproducible results
    np.random.seed(42)
    # Make a list to keep model scores
    model_scores = {}
    # Loop through models
    for name, model in models.items():
        # Fit the model to the data
        model.fit(X_train, y_train)
        # Evaluate the model and append its score to model_scores
        model_scores[name] = model.score(X_test, y_test)
    return model_scores


# In[28]:


model_scores= fit_and_score(models= models, X_train= X_train ,X_test= X_test,y_train=y_train,y_test=y_test)

model_scores


# ### Model Comparison 

# In[29]:


model_compare = pd.DataFrame(model_scores, index=["accuracy"])

model_compare.T.plot.bar() #T is transpose


# We now have the baseline model and a model's first predictions aren't always what we should based our next steps off.
# 
# 1. Hyperparameter tuning - Each model you use has a series of dials you can turn to dictate how they perform. Changing these values may increase or decrease model performance.
# 2. Feature importance - If there are a large amount of features we're using to make predictions, do some have more importance than others? For example, for predicting heart disease, which is more important, sex or age?
# 3. Confusion matrix - Compares the predicted values with the true values in a tabular way, if 100% correct, all values in the matrix will be top left to bottom right (diagnol line).
# 4. Cross-validation - Splits your dataset into multiple parts and train and tests your model on each part and evaluates performance as an average.
# 5. Precision - Proportion of true positives over total number of samples. Higher precision leads to less false positives.
# 6. Recall - Proportion of true positives over total number of true positives and false negatives. Higher recall leads to less false negatives.
# 7. F1 score - Combines precision and recall into one metric. 1 is best, 0 is worst.
# 8. Classification report - Sklearn has a built-in function called classification_report() which returns some of the main classification metrics such as precision, recall and f1-score.
# 9. ROC Curve - Receiver Operating Characterisitc is a plot of true positive rate versus false positive rate.
# 10. Area Under Curve (AUC) - The area underneath the ROC curve. A perfect model achieves a score of 1.

# ## Hyperparameter tuning(by hand)
# 

# In[30]:


# Create a list of train scores
train_scores = []

# Create a list of test scores
test_scores = []

# Create a list of different values for n_neighbors
neighbors = range(1, 21) # 1 to 20

# Setup algorithm
knn = KNeighborsClassifier()

# Loop through different neighbors values
for i in neighbors:
    knn.set_params(n_neighbors = i) # set neighbors value
    
    # Fit the algorithm
    knn.fit(X_train, y_train)
    
    # Update the training scores
    train_scores.append(knn.score(X_train, y_train))
    
    # Update the test scores
    test_scores.append(knn.score(X_test, y_test))


# In[31]:


train_scores


# In[32]:


test_scores


# In[33]:


plt.plot(neighbors, train_scores, label= "Train scores" )

plt.plot(neighbors,test_scores, label= "Test score ")

plt.xlabel("Number of neighbors ")
plt.xticks(np.arange(1, 21, 1))

plt.ylabel("Model score")

plt.legend()

print(f"Maximum KNN score on the test data: {max(test_scores)*100:.2f}%")


# ## Hyperparameter tuning with RandomizedSearchCV
# 
# We're going to tune
# 
# * LogisticRegression()
# * RandomForestClassifier()

# In[34]:


# Create a hyperparameter grid for LogisticRegression
log_reg_grid = {"C": np.logspace(-4, 4, 20),
                "solver": ["liblinear"]}

# Create a hyperparameter grid for RandomForestClassifier
rf_grid = {"n_estimators": np.arange(10, 1000, 50),
           "max_depth": [None, 3, 5, 10],
           "min_samples_split": np.arange(2, 20, 2),
           "min_samples_leaf": np.arange(1, 20, 2)}


# Now we've got hyperparameter grids setup for each of our models, let's tune them using RandomizedSearchCV...

# In[35]:


#Tune LogisticRegression

np.random.seed(42)

#Setup random hyperparameter search for LogisticRegression 

rs_log_reg = RandomizedSearchCV(LogisticRegression(),param_distributions=log_reg_grid,
                                cv=5,
                                n_iter=20,
                                verbose=True)

# Fit random hyperparameter search model for LogisticRegression
rs_log_reg.fit(X_train, y_train)


# In[36]:


rs_log_reg.best_params_


# In[37]:


rs_log_reg.score(X_test,y_test)


# Now we've tuned LogisticRegression(), let's do the same for RandomForestClassifier()...

# In[38]:


# Setup random seed
np.random.seed(42)

# Setup random hyperparameter search for RandomForestClassifier
rs_rf = RandomizedSearchCV(RandomForestClassifier(), 
                           param_distributions=rf_grid,
                           cv=5,
                           n_iter=20,
                           verbose=True)

# Fit random hyperparameter search model for RandomForestClassifier()
rs_rf.fit(X_train, y_train)


# In[39]:


# Find the best hyperparameters
rs_rf.best_params_


# In[40]:


# Evaluate the randomized search RandomForestClassifier model
rs_rf.score(X_test, y_test)


# ## Hyperparameter tuning using GridSearchCV
# 
# Since our LogisticRegression model provides the best scores so far , we'll try and improve them again using GridSearchCV...

# In[41]:


#Different hyperparameters for our LogisticRegression 

log_reg_grid= {"C": np.logspace(-4,4,30),
              "solver":["liblinear"]}

#Setup grid parameters for LogisticRegression

gs_log_reg = GridSearchCV(LogisticRegression(), param_grid=log_reg_grid,cv=5,verbose=True) #no n-iter as GSCV tries every combo

#Fit grid hyperparameter search model

gs_log_reg.fit(X_train ,y_train);


# In[42]:


#check the best params

gs_log_reg.best_params_


# In[43]:


#Evaluate the gridsearch logistic regression model

gs_log_reg.score(X_test,y_test)


# In[44]:


model_scores


# ## Evaulating our tuned machine learning classifier , beyond accuracy 
# 
# * ROC curve and AUC score
# * Confusion Matrix 
# * Classification report
# * Precision 
# * Recall 
# * F-1 score
# 
# It would be great if cross validation is used wherever possible .
# 
# To make comparisons and evaluate our trained model , first we need to make predictions 

# In[45]:


#Make predictions with tuned model 

y_preds = gs_log_reg.predict(X_test)


# In[46]:


y_preds


# In[47]:


y_test


# In[48]:


#Plot ROC curve and calculate AUC metric 

plot_roc_curve(gs_log_reg, X_test ,y_test)


# In[49]:


#Confusion matrix

print(confusion_matrix(y_test,y_preds))


# In[50]:


sns.set(font_scale=1.5) # Increase font size
 
def plot_conf_mat(y_test, y_preds):
    """
    Plots a confusion matrix using Seaborn's heatmap().
    """
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.heatmap(confusion_matrix(y_test, y_preds),
                     annot=True, # Annotate the boxes
                     cbar=False)
    
plot_conf_mat(y_test, y_preds)


# Now we've got a ROC curve , AUC metric and a confusion matrix . Let's get a classification report as well as cross validated precision , recall and f1 score 

# In[51]:


print(classification_report(y_test,y_preds)) #Train test split (1 split)


# ### Calculate evaluation metrics using cross validation 
# 
# we are going to calculate accuracy precision, recall and f1 score of our model using cross validation and to do so we'll be using 
# `cross_val_score()`

# In[52]:


gs_log_reg.best_params_


# In[53]:


#Create a new classifier with best parameters

clf = LogisticRegression(C=0.20433597178569418 , solver= 'liblinear' )


# In[54]:


#cross validated accuracy 

cv_acc = cross_val_score(clf, X ,y, cv=5 , scoring = "accuracy")


# In[55]:


cv_acc =  np.mean(cv_acc)
cv_acc


# In[56]:


#Cross validated precision

cv_precision = cross_val_score(clf, X ,y, cv=5 , scoring = "precision")

cv_precision= np.mean(cv_precision)

cv_precision


# In[57]:


#cross validated recall

cv_recall = cross_val_score(clf, X ,y, cv=5 , scoring = "recall")

cv_recall= np.mean(cv_recall)

cv_recall


# In[58]:


#cross validated f1 score 

#cross validated recall

cv_f1 = cross_val_score(clf, X ,y, cv=5 , scoring = "f1")

cv_f1= np.mean(cv_f1)

cv_f1


# In[59]:


#visualize our cross validated metrics( making a function is better )

cv_metrics = pd.DataFrame({"Accuracy": cv_acc, "Precision": cv_precision, "Recall": cv_recall, "f1": cv_f1 },index=[0]) 

cv_metrics.T.plot.bar(title= "Cross-validated classification metrics", legend="False");


# ### Feature importance 
# 
# Feature importance is another way of asking "which features contributed most to the outcomes and how did they contribute? "
# Finding feature importance is different for each machine learning model.One way to find feature importance is to search for (MODEL NAME) feature importance".
# 
# Let's find the feature importance for our logistic regression model

# In[60]:


#fit an instance of logistic regression

#gs_log_reg.best_params_

clf= LogisticRegression(C=0.20433597178569418, solver= "liblinear")

clf.fit(X_train, y_train);


# In[61]:


#check coef_

clf.coef_


# In[62]:


#Match the coef's of features to column 

feature_dict= dict(zip(df.columns,list(clf.coef_[0])))
feature_dict


# In[63]:


df.head()


# In[64]:


#Visualize feature importance 

feature_df = pd.DataFrame(feature_dict, index=[0])

feature_df.T.plot.bar(title= "Feature importance" , legend =False);


# In[66]:


pd.crosstab(df["sex"], df["target"])


# In[68]:


pickle.dump(clf, open('model.pkl', 'wb'))


# In[69]:


model = pickle.load(open('model.pkl', 'rb'))


# In[70]:


df.head()


# In[ ]:





# In[73]:


print(model.predict([[55,1,2,125,240,0,1,166,1,2.0,0,0,1]]))


# In[ ]:





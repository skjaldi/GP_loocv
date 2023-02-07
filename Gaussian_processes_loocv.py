#!/usr/bin/env python
# coding: utf-8

# ### Hei, Simen! 
# 
# We'll be using the following packages:
# 

# In[1]:


import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
from sklearn.metrics import r2_score


# The dataframe we'll be using has been refined (f.x. the groupby-operation has already been applied). It's the same one that gave an R-squared last semester of around 0.4. 

# In[2]:


df = pd.read_pickle("df_loocv.pkl")

df.head(5)


# As you can see, it consists of Amine names, where each row is a specific amine. The rest are columns with numerical values, which will be used to build a model.
# 
# 
# "<b>MW</b>" is molar weight, 
# 
# "<b>loading_mol_mol_nitrogen</b>" is CO2 loading per amine group at 40°C,
# 
# "<b>pka_plot</b>" is the pka-value,
# 
# and "<b>cyclic_mol_mol_nitrogen</b>" is the cyclic capacity per amine group in the range 40-80°C.

# Let's look at the length of df:

# In[3]:


len(df)


# That is, we've got 48 datapoints

# Let's now drop the amine column, to use df for model-building:

# In[4]:


df = df.drop(["Amines"], axis=1)


# We'll be using Leave-one-out-cross-validation (LOOCV) to get an R2 score. To the best of my knowledge, it is not possible to use scikit or other packages for that purpose, since they need more than one datapoint to calculate an R2 score. The best solution I could find was to do a for-loop through df.
# 
# For each datapoint (or row) in df, we will do an iteration, where that row will be saved and then dropped from df. Then, df will be used to train a model. And finally, the target of the left-out row  will be appended outside the loop, along with the model prediction, based on the input variables from the left out row. 
# 
# The left-out values will be stored in the list "ytests", whereas the values that the model predicts will be stored in the list "ypreds".
# 
# These lists will at the end then be used to calculate an R2 score.
# 
# The target (what we want to predict) is in this case the cyclic capcaity.
# 
# Note that the code below will take a few seconds to run, since it's training 48 models. I've tried to include a few comments within the code. Also note that you might get a few ConvergenceWarnings, which I don't get in my usual Spyder environment. This is something I'd maybe like to talk to you about.

# In[5]:


ytests = []
ypreds = []

target = "cyclic_mol_mol_nitrogen"

for i in range(len(df)):
    
    df_loop = df  # Since we don't want to change the original dataframe
    
    test_line = df_loop.iloc[[i]]                    # This is the row that will be excluded when training the model
    y_test = test_line[target].values.reshape(-1,1)  # Save the value of the actual cyclic capacity
    X_test = test_line.drop([target], axis=1)        # Save the input values (everything except cyclic capcaity)

    df_loop = df_loop.drop([df_loop.index[i]])   # drop the row of index i
    
    y = df_loop[target]                  # Our model-training targets
    X = df_loop.drop([target], axis=1)   # Our model-training inputs
    
    X_train, y_train = X, y   # This is more so just a formality
    
    kernel = DotProduct() + WhiteKernel() + RBF()   # The combination of these kernels seem to work well
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5) ### n_restarts_optimizer is the amount
    gaussian_process.fit(X_train, y_train)                                              ## of times that the optimizer will restart
                                                                                         # to optimze the kernel hyperparameters
    
    ytests.append(y_test[0][0])                     # We append the actual value
    y_predicted = gaussian_process.predict(X_test)  # Make a prediction from the inputs
    ypreds.append(y_predicted[0])                   # And append the prediction value
    


# Now we can finally calculate the R2 score with the ytests and ypreds lists.

# In[6]:


r_scores = r2_score(ytests, ypreds)   

r_scores


# And as you can see, we got a score of 0.58, which is quite a bit better than than the models last semester.

# To showcase how predictions are made, with a standard deviation, let's quickly first make a model using all the data (excluding no row in df), in a similar fashion as in the for-loop. Note that you might get the ConvergenceWarning again...

# In[7]:


df2 = df
y = df2[target]
X = df2.drop([target], axis=1)

X_train, y_train = X, y

kernel = DotProduct() + WhiteKernel() + RBF()
gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=4) 
gaussian_process.fit(X_train, y_train)


# Let's say that we are hypothetically examining a new amine solvent that is one percent larger than MEA in all input values, and we want to predict what the cyclic capacity might be for that solvent. Note the values for MEA:

# In[8]:


df.loc[0]


# Let's then make the prediction:

# In[9]:


prediction_mean, prediction_std = gaussian_process.predict([[61.69, 0.549839, 9.53]], return_std=True)

print(f"Which gives a mean prediction of {prediction_mean[0]:.3f}, with a standard deviation of {prediction_std[0]*1.96:.3f}")


# Now, the standard deviation might seem large, but the mean prediction doesn't seem so far-fetched for our hypothetical new solvent. The R2 score we got (0.58) was from the mean (or average) predictions. The standard deviation tells us the 95% confidence interval.
# 
# And I think I've said all I wanted to say :)

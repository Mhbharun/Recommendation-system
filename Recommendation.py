
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



# Importing packages 

# In[2]:


dataset= pd.read_csv("data.csv") 


# In[3]:


dataset


# Own dataset has been created for this problem
# 
# 1.Dataset is read using pandas library in which the list of the courses are collected from different websites such as couresera and udemy etc.
# 2. The Dataset contains ID and List of courses
# 3.First 10 ID are taken as a reference input for the given problem

# In[4]:


tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 4), min_df=0, stop_words='english')


# In[6]:


tfidfmat = tf.fit_transform(dataset['Courses'])
cosinesimilarity = cosine_similarity(tfidfmat,tfidfmat)

results = {} 


# In[7]:


tfidfmat


# In[8]:


cosinesimilarity


# In[9]:


for idx, row in dataset.iterrows(): 
    similarindex = cosinesimilarity[idx].argsort()[:-10:-1] 
    similaritems = [(cosinesimilarity[idx][i], dataset['ID'][i]) for i in similarindex]
    results[row['ID']] = similaritems[1:]


# In[10]:


similarindex


# In[11]:


similaritems


# In[12]:



def item(id):
    return dataset.loc[dataset['ID'] == id]['Courses'].tolist()[0]
def userinput(id, num):
    if (num == 0):
        print("Unable to recommend any Courses")
    elif (num==1):
        print("Recommend " + str(num) + " Courses similar to " + item(id))
        
    else :
        print("Recommend" + str(num) + " Courses similar to " + item(id))
        
    print("----------------------------------------------------------")
    recs = results[id][:num]
    for rec in recs:
        print("You may also like to read: " + item(rec[1]) + " (score:" + str(rec[0]) + ")")


# In[13]:



userinput(8,6)


# from the dataset ID from 1-10 is taken as an input
# 
# From userinput(6,3)
# 6 is the input for the ID
# 3 represents the number of courses to be listed 
# 
# from the list score derives the best book for the given keyword
# if score is zero then it doest match with the given input keyword

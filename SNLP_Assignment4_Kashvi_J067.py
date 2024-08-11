#!/usr/bin/env python
# coding: utf-8

# In[7]:


import transformers


# In[34]:


tokenizer = transformers.BertTokenizer.from_pretrained("google-bert/bert-base-uncased") #tokenizerid


# In[8]:


tokenizer = transformers.AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")


# In[9]:


tokenizer = transformers.AutoTokenizer.from_pretrained("FacebookAI/roberta-base", use_fast=True)


# In[10]:


text= "I am Kashvi and I am studying Data Science"

tokenizer.tokenize(text)


# In[8]:


##in roberta the space is g. in bert the word is tokenized as it is


# In[11]:


enc = tokenizer(text)


# In[12]:


for id in enc["input_ids"]:
    print(id, tokenizer.decode(id))


# In[13]:


enc = tokenizer(
    text,
    max_length=50, 
    padding = 'max_length', 
    truncation=True,
    return_offsets_mapping=True,
)


# In[14]:


#padding is done to ensure same size for all our inputs


# In[15]:


enc


# In[16]:


#if attention mask is 0; not relevant. input and attention will always be of the same size. 
#if padding hadnt been done, the attention wouldnt have updated 


# In[17]:


enc.keys()


# In[18]:


enc.word_ids()


# In[ ]:





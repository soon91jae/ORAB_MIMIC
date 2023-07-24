#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] ="4"

import random 
random.seed(0)

import collections
from collections import Counter

from itertools import islice, chain
import gc


# In[2]:


import pickle

import tqdm
#import tqdm.notebook as tqdm

from sklearn.model_selection import KFold

# In[3]:


import numpy as np


# In[4]:


import torch
from torch import nn
import os
device = 'cuda' if torch.cuda.is_available() else 'cpu'



# In[5]:


torch.cuda.current_device()


# In[6]:


import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel, AutoModelForPreTraining, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoConfig
from transformers import pipeline


# In[7]:




# In[8]:


#raw_text_dict[data[0][-1]]


# In[9]:


#MODEL_PATH = "/data/python_envs/anaconda3/envs/transformers_cache/Bio_ClinicalBERT"
MODEL_PATH = "/data/python_envs/anaconda3/envs/transformers_cache/Bio_ClinicalBERT"
#MODEL_PATH = "/data/python_envs/anaconda3/envs/transformers_cache/biobert-v1.1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


# In[10]:


bz = 8
epochs = 3
lr = 2e-5
folds = 5
multi_task = True

print("bz:%d, epochs:%d, lr: %e, Multitask; %s"%(bz,epochs, lr, multi_task))

data_path = '/home/vhabedkwons/Projects/Abberant_Behavior/dataset/extract_ehost_MIMIC/extract_ehost_UMass/extract_ehost/'
data_path = '/home/vhabedkwons/Projects/Abberant_Behavior/dataset/extract_ehost/'

data = pickle.load(open(data_path+'pkls.pkl','rb'))
raw_text_dict = pickle.load(open(data_path+'raw_text_dict.pkl','rb'))
# In[11]:


data_dict = {}
sent_dict= {}

tag_cnt = Counter()
tag2id = {}
id2tag = {}
ABtag2id = {}
id2ABtag = {}


# In[12]:


temp_sent_id = 0
for example in data:
    #print(example)
    start, end, instance, tag, context, document = example
    
    #if 'ABERRANT_BEHAVIOR' not  in tag: continue;
    #tag = 'ABERRANT_BEHAVIOR'
    
    instance = instance.strip()
    context = context.strip()
    
    
    if context not in sent_dict:
        sent_dict[context] = temp_sent_id
        temp_sent_id += 1
        
        sent_id = sent_dict[context]
        data_dict[sent_id] = {}
        data_dict[sent_id]['context'] = context
        data_dict[sent_id]['instances'] = []
    
    sent_id = sent_dict[context]    
    #match = (re.search(instance, context))
    start_index = context.find(instance)
    if start_index != -1:
        instance_dict = {'start': start_index,
                         'end': start_index + len(instance),
                         'instance': instance,
                         'tag': tag}
        data_dict[sent_id]['instances'].append(instance_dict)
    else:
        print(context, instance)
    
    #if tag not in tag2id and 'ABERRANT_BEHAVIOR' not in tag: 
    if tag not in tag2id: 
        if not multi_task and 'ABERRANT_BEHAVIOR' not in tag: continue 
        tag2id[tag] = len(tag2id)
    if tag not in ABtag2id and 'ABERRANT_BEHAVIOR' in tag: 
        ABtag2id[tag] = len(ABtag2id)
    tag_cnt[tag] += 1
for tag in tag2id.keys(): 
    id2tag[tag2id[tag]] = tag

#print(ABtag2id)
ABtag2id['none'] = len(ABtag2id)

for tag in ABtag2id.keys():
    id2ABtag[ABtag2id[tag]] = tag


# In[13]:


# for key in data_dict.keys():
#     tags = []
#     for instance in data_dict[key]['instances']:
#         tags.append(instance['tag'])
#     if len(set(tags))>1:
#         print(data_dict[key])
# #data_dict[11]


# In[14]:


id2tag


# In[15]:


id2ABtag


# In[16]:


#data_dict[key]


# In[17]:


# print(len(data_dict))

# instance_num = 0
# for key in data_dict.keys():
#     instance_num += len(data_dict[key]['instances'])
# instance_num


# In[18]:


sent_keys = list(data_dict.keys())
random.shuffle(sent_keys)


kf = KFold(n_splits = folds)
kf.get_n_splits(sent_keys)

for i, (train_index, test_index) in enumerate(kf.split(sent_keys)):
    
    train_keys = [sent_keys[index] for index in train_index]
    #dev_keys = sent_keys[int(len(sent_keys) * 0.8):int(len(sent_keys) * 0.9)]
    test_keys = [sent_keys[index] for index in test_index]

    train_data = [data_dict[key] for key in train_keys]
    dev_data = [data_dict[key] for key in test_keys]
    test_data = [data_dict[key] for key in test_keys]


    # In[19]:


    #train_data[0]


    # In[20]:




    # In[21]:


    class data_processor(object):
        def __init__(self, tag2id, ABtag2id, tokenizer, multi_task=True):
            self.tag2id = tag2id
            self.ABtag2id = ABtag2id
            self.tokenizer = tokenizer
            self.multi_task = multi_task
            return


        def processing(self, data):
            tag2id = self.tag2id
            ABtag2id = self.ABtag2id
            tokenizer = self.tokenizer
            multi_task = self.multi_task
            processed_data = []

            SEP_token = tokenizer.sep_token
            MASK_token = tokenizer.mask_token
            template = "The type of aberrant behavior is"

            for example in tqdm.tqdm(data):
                prompt = " %s %s %s"%(SEP_token, template, MASK_token)

                multitask_prompt = " %s "%(SEP_token)

                for tag in list(tag2id.keys()):
                    if tag == 'MED CHANGE': tag = 'medication change'
                    multitask_prompt = multitask_prompt + "%s %s "%(tag, MASK_token)
                multitask_prompt = multitask_prompt + "%s %s"%(template, MASK_token)        

                context = example['context'];
                multitask_prompt = context + multitask_prompt
                context_prompt = context + prompt

                instances = example['instances']
                input_ids = tokenizer.encode(context)
                prompting_ids = tokenizer.encode(context_prompt)

                #print(context_prompt)
                #print(multitask_prompt)
                tags = [0.0] * len(tag2id)
                for instance in instances:
                    tag = instance['tag']
                    if tag in tag2id:
                        tags[tag2id[tag]] = 1.0

                ABtags = [0] * len(ABtag2id)
                for instance in instances:
                    tag = instance['tag']
                    if tag in ABtag2id:
                        ABtags[ABtag2id[tag]] = 1
                if sum(ABtags) == 0: ABtags[ABtag2id['none']] = 1
                elif sum(ABtags) == 2: ABtags[ABtag2id['SUGGEST_ABERRANT_BEHAVIOR']] = 0
                ABtags = np.argmax(ABtags)


                example_dict = {"context": context,
                                "input_ids": input_ids,
                                "context_prompt": context_prompt,
                                #"prompting_ids": prompting_ids,
                                "mutitask_prompt": multitask_prompt,
                                #"mutitask_prompt_ids": mutitask_prompt_ids,
                                "tags": tags,
                                "ABtags": ABtags}
                processed_data.append(example_dict)

            return processed_data


    # In[22]:


    processor = data_processor(tag2id, ABtag2id, tokenizer, multi_task)

    train_inputs = processor.processing(train_data)
    dev_inputs = processor.processing(dev_data)
    test_inputs = processor.processing(test_data)


    # In[23]:


    # train_inputs[0]


    # In[24]:


    tag2id, ABtag2id


    # In[25]:


    # import pandas as pd

    # data = {tag:[] for tag in tag2id.keys()}
    # for train_input in train_inputs:
    #     tags = train_input['tags']

    #     for tag_id, tag in enumerate(tags):

    #         data[id2tag[tag_id]].append(tag)
    # df = pd.DataFrame(data)


    # In[26]:


    # from scipy.stats import chi2_contingency

    # def cramers_V(var1, var2):
    #     crosstab = np.array(pd.crosstab(var1, var2))
    #     stat = chi2_contingency(crosstab)[0]
    #     obs = np.sum(crosstab)
    #     mini = min(crosstab.shape)-1
    #     return (stat/(obs*mini))


    # In[27]:


    # import itertools

    # chi2dict = {}
    # for col1, col2 in itertools.permutations(df.columns,2):
    #     print(col1, col2, cramers_V(df[col1], df[col2]))


    # In[28]:


    # for test_input in test_inputs:
    #     context = test_input['context']
    #     tags = test_input['tags']

    #     if tags[tag2id['MED CHANGE']] == 1.0:
    #         print(context)


    # In[ ]:





    # In[29]:


    class batchfier(object):
        def __init__(self, inputs, tokenizer, id2tag, bz=32, shuffle=True):
            self.inputs =  inputs#inputs
            self.bz = bz
            self.shuffle = shuffle
            self.tokenizer = tokenizer
            self.id2tag = id2tag
        def get_batches(self):
            inputs= self.inputs
            bz = self.bz
            shuffle = self.shuffle
            tokenizer = self.tokenizer
            id2tag = self.id2tag

            def chunks(lst,n):
                for i in range(0,len(lst),n):
                    yield lst[i:i+n]

            #total_batch_num = int(len(inputs)/bz)+1
            input_indexs = list(range(len(inputs)))
            if shuffle:
                random.shuffle(input_indexs)
            batch_indexs = chunks(input_indexs, bz)


            batch_inputs = []
            for batch_index in batch_indexs:
                contexts = [inputs[index]['context'] for index in batch_index]
                context_prompt = [inputs[index]['context_prompt'] for index in batch_index]
                multitask_prompt = [inputs[index]['mutitask_prompt'] for index in batch_index]
                original_inputs = tokenizer.batch_encode_plus(contexts,  pad_to_max_length=True)
                prompt_inputs = tokenizer.batch_encode_plus(context_prompt,  pad_to_max_length=True)
                multitask_prompt_inputs = tokenizer.batch_encode_plus(multitask_prompt,  pad_to_max_length=True)
                #multitask_prompt_inputs = tokenizer.batch_encode_plus()


                batch_input_ids = prompt_inputs['input_ids']
                prompt_mask = [[0.0]*len(batch_input_ids[0]) for index in batch_index]

                tags = [inputs[index]['tags'] for index in batch_index]
                
                   
                try:
                    #print(len(batch_input_ids[1]))
                    #print(len(batch_input_ids[0]))
                    ABtags = [[inputs[index]['ABtags']] * len(batch_input_ids[0]) for index in batch_index]
                except: 
                    print(inputs[index])
                    print(batch_input_ids)
                    exit()
                #auxtags = [[inputs[index]['tags']] * len(batch_input_ids[1]) for index in batch_index]

                for i, input_ids in enumerate(batch_input_ids):
                    for j, input_id in enumerate(input_ids):
                        if batch_input_ids[i][j] == tokenizer.mask_token_id:
                            prompt_mask[i][j] = 1.0

                batch_input_ids = multitask_prompt_inputs['input_ids']
                multitask_prompt_mask = [[0.0]*len(batch_input_ids[0]) for index in batch_index]
                auxiliary_prompt_mask = [[0.0]*len(batch_input_ids[0]) for index in batch_index]

                multitask_Auxtags = np.zeros((len(batch_index), len(batch_input_ids[0])))
                multitask_ABtags = [[inputs[index]['ABtags']] * len(batch_input_ids[0]) for index in batch_index]
                for i, input_ids in enumerate(batch_input_ids):
                    mask_indexes = []
                    for j, input_id in enumerate(input_ids):
                        if batch_input_ids[i][j] == tokenizer.mask_token_id:
                            mask_indexes.append(j)

                    for index, tag in zip(mask_indexes[:-1], tags[i]):
                        auxiliary_prompt_mask[i][index] = 1.0
                        if tag == 1: multitask_Auxtags[i][index] = 1.0
                        #else: multitask_Auxtags[i][index][0] = 1.0
                    multitask_prompt_mask[i][mask_indexes[-1]] = 1.0

                            #multitask_prompt_mask[i][j] = 1.0
                multitask_Auxtags = multitask_Auxtags.astype(int)

                batch_input = {'tags': tags,
                               'ABtags': ABtags,
                               'multitask_ABtags': multitask_ABtags,
                               'multitask_Auxtags': multitask_Auxtags,
                               'contexts': contexts,
                               'inputs': original_inputs,
                               'context_prompt': context_prompt, 
                               'prompt_inputs': prompt_inputs,
                               'prompt_mask': prompt_mask,
                               'multitask_prompt': multitask_prompt,
                               'multitask_prompt_inputs': multitask_prompt_inputs,
                               'multitask_prompt_mask': multitask_prompt_mask,
                               'auxiliary_prompt_mask': auxiliary_prompt_mask}

                batch_inputs.append(batch_input)
            return batch_inputs   


    # In[30]:


    train_batches = batchfier(train_inputs, tokenizer, id2tag, bz)
    test_batches = batchfier(test_inputs, tokenizer, id2tag, bz, shuffle=False)
    dev_batches = batchfier(dev_inputs, tokenizer, id2tag, bz, shuffle=False)


    # In[31]:


    # batch_examples = dev_batches.get_batches()


    # In[32]:


    # index = 13

    # print(batch_examples[0]['auxiliary_prompt_mask'][index])
    # print(batch_examples[0]['multitask_prompt_mask'][index])
    # print(batch_examples[0]['multitask_prompt'][index])
    # print(batch_examples[0]['multitask_ABtags'][index])
    # print(batch_examples[0]['multitask_Auxtags'][index])
    # print(batch_examples[0]['tags'][index])


    # In[ ]:





    # In[33]:


    class BaselineModel(torch.nn.Module):
        def __init__(self, MODEL_PATH, tag2id, ABtag2id):
            super(BaselineModel, self).__init__()

            self.plm = AutoModel.from_pretrained(MODEL_PATH)
            self.config = AutoConfig.from_pretrained(MODEL_PATH)
            self.dropout = torch.nn.Dropout(0.3)
            self.classifier = torch.nn.Linear(self.config.hidden_size, len(tag2id))
            self.ABclassifier = torch.nn.Linear(self.config.hidden_size, len(ABtag2id))
            self.Auxclassifier = torch.nn.Linear(self.config.hidden_size, 2)

        def forward(self, input_ids, token_type_ids, attention_mask):
            _, hiddens = self.plm(input_ids, token_type_ids, attention_mask)
            outputs = self.dropout(hiddens)
            logits = self.classifier(outputs)
            ABlogits = self.ABclassifier(outputs)
            Auxlogits = self.Auxclassifier(outputs)


            return logits, ABlogits, Auxlogits, hiddens


    # In[34]:


    print(tokenizer.encode('none',add_special_tokens=None))
    print(tokenizer.encode('confirm',add_special_tokens=None))
    print(tokenizer.encode('suggest',add_special_tokens=None))


    # In[ ]:





    # In[35]:


    # plm = AutoModelForPreTraining.from_pretrained(MODEL_PATH)


    # In[36]:


    # inputs = tokenizer.batch_encode_plus(['I love you [MASK],','you love me [MASK]'], pad_to_max_length=True)


    # In[37]:


    # input_ids = torch.tensor(inputs['input_ids'])
    # token_type_ids = torch.tensor(inputs['token_type_ids'])
    # attention_mask = torch.tensor(inputs['attention_mask'])


    # In[38]:


    # output_logits = plm(input_ids, token_type_ids, attention_mask)[0]
    # output_logits.shape


    # In[39]:


    #masked_logits = output_logits[:,:,[12434, 5996, 3839]]
    #masked_logits = masked_logits.permute(1,0,2)


    # In[ ]:





    # In[40]:


    #tokenizer.encode('confirm',add_special_tokens=None)


    # In[41]:


    class PromptBasedModel(torch.nn.Module):
        def __init__(self, MODEL_PATH, tag2id, ABtag2id):
            super(PromptBasedModel, self).__init__()

            self.plm = AutoModelForPreTraining.from_pretrained(MODEL_PATH)
            self.config = AutoConfig.from_pretrained(MODEL_PATH)
    #         self.dropout = torch.nn.Dropout(0.3)
    #         self.classifier = torch.nn.Linear(self.config.hidden_size, len(tag2id))
    #         self.ABclassifier = torch.nn.Linear(self.config.hidden_size, len(ABtag2id))

            # 
    #         tokenizer.encode('none',add_special_tokens=None)
    #         tokenizer.encode('confirm',add_special_tokens=None)
    #         tokenizer.encode('suggest',add_special_tokens=None)
            none_word_index = tokenizer.encode('none',add_special_tokens=None)[0]
            confirm_index = tokenizer.encode('confirm',add_special_tokens=None)[0]
            suggest_index = tokenizer.encode('suggest',add_special_tokens=None)[0]
            self.class_indexes = [confirm_index, suggest_index, none_word_index]


            no_word_index = tokenizer.encode('no',add_special_tokens=None)[0]
            yes_word_index = tokenizer.encode('yes',add_special_tokens=None)[0]
            self.aux_calss_indexes = [no_word_index, yes_word_index]
        def forward(self, input_ids, token_type_ids, attention_mask, ABtags = None, prompt_mask = None):
            outputs = self.plm(input_ids, token_type_ids, attention_mask)
            #outputs = self.dropout(hiddens)

            output_logits = outputs[0]
            prompt_logits = output_logits[:,:,self.class_indexes]
            aux_logits = output_logits[:,:,self.aux_calss_indexes]
            #logits = self.classifier(outputs)
            #ABlogits = self.ABclassifier(outputabss)
            #print(masked_logits.shape)


            return prompt_logits, aux_logits


    # In[42]:


    model = PromptBasedModel(MODEL_PATH, tag2id, ABtag2id)
    model = model.to(device)


    # In[ ]:





    # In[43]:


    # AutoConfig.from_pretrained(MODEL_PATH).hidden_size
    total_step = len(train_batches.get_batches())*epochs

    optim = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optim, 
                                                num_warmup_steps=total_step*0.1,
                                                num_training_steps=total_step)

    criterion = nn.CrossEntropyLoss()
    aux_criterion = nn.BCEWithLogitsLoss()


    # In[44]:


    def cross_entropy_mask(masked_logits, ABtags, prompt_mask):
        #loss = None
        if ABtags is not None:
            loss_fct = nn.CrossEntropyLoss()

            active_loss = prompt_mask.view(-1) == 1
            active_logits = masked_logits.view(-1, masked_logits.shape[-1])
            active_labels = torch.where(
                active_loss, ABtags.view(-1), torch.tensor(loss_fct.ignore_index).type_as(ABtags)
            )
            loss = loss_fct(active_logits, active_labels)
        return loss


    # In[45]:


    losses = []
    best_model = None
    best_loss = None
    for epoch in tqdm.tqdm(range(epochs)):
        gc.collect()
        torch.cuda.empty_cache()
        for train_batch in tqdm.tqdm(train_batches.get_batches(), leave=True):

            optim.zero_grad()

#             if not multi_task:
#                 tags = torch.tensor(train_batch['tags']).to(device)
#                 ABtags = torch.tensor(train_batch['ABtags']).to(device)
#                 inputs = train_batch['prompt_inputs']

#                 input_ids = torch.tensor(inputs['input_ids']).to(device)
#                 token_type_ids = torch.tensor(inputs['token_type_ids']).to(device)
#                 attention_mask = torch.tensor(inputs['attention_mask']).to(device)

#                 prompt_mask = torch.tensor(train_batch['prompt_mask']).to(device)

#                 ABlogits, _ = model(input_ids, token_type_ids, attention_mask, ABtags, prompt_mask)
#                 loss = cross_entropy_mask(ABlogits, ABtags, prompt_mask)
#                 losses.append(loss.detach().cpu().numpy())
#             else:
            multitask_ABtags = torch.tensor(train_batch['multitask_ABtags']).to(device)
            multitask_Auxtags = torch.tensor(train_batch['multitask_Auxtags']).to(device)

            inputs = train_batch['multitask_prompt_inputs']
            input_ids = torch.tensor(inputs['input_ids']).to(device)
            token_type_ids = torch.tensor(inputs['token_type_ids']).to(device)
            attention_mask = torch.tensor(inputs['attention_mask']).to(device)

            multitask_prompt_mask = torch.tensor(train_batch['multitask_prompt_mask']).to(device)
            auxiliary_prompt_mask = torch.tensor(train_batch['auxiliary_prompt_mask']).to(device)

            ABlogits, Auxlogits = model(input_ids, token_type_ids, attention_mask)
            loss = cross_entropy_mask(ABlogits, multitask_ABtags, multitask_prompt_mask)
            #losses.append(loss.detach().cpu().numpy()) 

            aux_loss = cross_entropy_mask(Auxlogits, multitask_Auxtags, auxiliary_prompt_mask)                        
            loss = aux_loss

            losses.append(loss.detach().cpu().numpy())                                                                                                                                            

            loss.backward()
            optim.step()
            scheduler.step()
            #probs = torch.sigmoid(logits)

        test_losses=[]
        dev_losses = []
        with torch.no_grad():
            for dev_batch in dev_batches.get_batches():
                #print(train_batch)
                optim.zero_grad()

#                 if not multi_task:
#                     tags = torch.tensor(dev_batch['tags']).to(device)
#                     ABtags = torch.tensor(dev_batch['ABtags']).to(device)
#                     inputs = dev_batch['prompt_inputs']

#                     input_ids = torch.tensor(inputs['input_ids']).to(device)
#                     token_type_ids = torch.tensor(inputs['token_type_ids']).to(device)
#                     attention_mask = torch.tensor(inputs['attention_mask']).to(device)

#                     prompt_mask = torch.tensor(dev_batch['prompt_mask']).to(device)

#                     ABlogits, _ = model(input_ids, token_type_ids, attention_mask, ABtags, prompt_mask)
#                     loss = cross_entropy_mask(ABlogits, ABtags, prompt_mask)
#                     dev_losses.append(loss.cpu().detach().numpy())
#                 else:
                multitask_ABtags = torch.tensor(dev_batch['multitask_ABtags']).to(device)
                multitask_Auxtags = torch.tensor(dev_batch['multitask_Auxtags']).to(device)

                inputs = dev_batch['multitask_prompt_inputs']
                input_ids = torch.tensor(inputs['input_ids']).to(device)
                token_type_ids = torch.tensor(inputs['token_type_ids']).to(device)
                attention_mask = torch.tensor(inputs['attention_mask']).to(device)

                multitask_prompt_mask = torch.tensor(dev_batch['multitask_prompt_mask']).to(device)
                auxiliary_prompt_mask = torch.tensor(dev_batch['auxiliary_prompt_mask']).to(device)

                ABlogits, Auxlogits = model(input_ids, token_type_ids, attention_mask)
                loss = cross_entropy_mask(ABlogits, multitask_ABtags, multitask_prompt_mask)
                #losses.append(loss.detach().cpu().numpy()) 

                aux_loss = cross_entropy_mask(Auxlogits, multitask_Auxtags, auxiliary_prompt_mask)                                                                            
                loss = aux_loss

                dev_losses.append(loss.cpu().detach().numpy())

            for test_batch in test_batches.get_batches():
                #print(train_batch)
                optim.zero_grad()
#                 if not multi_task:
#                     tags = torch.tensor(test_batch['tags']).to(device)
#                     ABtags = torch.tensor(test_batch['ABtags']).to(device)
#                     inputs = test_batch['prompt_inputs']

#                     input_ids = torch.tensor(inputs['input_ids']).to(device)
#                     token_type_ids = torch.tensor(inputs['token_type_ids']).to(device)
#                     attention_mask = torch.tensor(inputs['attention_mask']).to(device)

#                     prompt_mask = torch.tensor(test_batch['prompt_mask']).to(device)

#                     ABlogits, _ = model(input_ids, token_type_ids, attention_mask, ABtags, prompt_mask)
#                     loss = cross_entropy_mask(ABlogits, ABtags, prompt_mask)
#                     test_losses.append(loss.cpu().detach().numpy())
#                 else:
                multitask_ABtags = torch.tensor(test_batch['multitask_ABtags']).to(device)
                multitask_Auxtags = torch.tensor(test_batch['multitask_Auxtags']).to(device)

                inputs = test_batch['multitask_prompt_inputs']
                input_ids = torch.tensor(inputs['input_ids']).to(device)
                token_type_ids = torch.tensor(inputs['token_type_ids']).to(device)
                attention_mask = torch.tensor(inputs['attention_mask']).to(device)

                multitask_prompt_mask = torch.tensor(test_batch['multitask_prompt_mask']).to(device)
                auxiliary_prompt_mask = torch.tensor(test_batch['auxiliary_prompt_mask']).to(device)

                ABlogits, Auxlogits = model(input_ids, token_type_ids, attention_mask)
                loss = cross_entropy_mask(ABlogits, multitask_ABtags, multitask_prompt_mask)
                #losses.append(loss.detach().cpu().numpy()) 

                aux_loss = cross_entropy_mask(Auxlogits, multitask_Auxtags, auxiliary_prompt_mask)                                                                            
                loss = aux_loss

                test_losses.append(loss.cpu().detach().numpy())
            print("%.2f, %.2f"%(np.mean(dev_losses), np.mean(test_losses)))




    # In[46]:


    len(test_batch)


    # In[47]:


    # ABtags.shape, prompt_mask.shape, input_ids.shape


    # In[48]:


    import matplotlib.pyplot as plt

    xpoints = list(range(len(losses)))
    plt.plot(xpoints, losses)
    plt.show()


    # In[49]:


    # !nvidia-smi


    # In[50]:


    test_losses=[]
    dev_losses = []

    test_probs = []
    test_tags = []
    test_aux_probs = []
    test_aux_tags = []
    test_context = []

    model.eval()

    with torch.no_grad():
        for dev_batch in tqdm.tqdm(dev_batches.get_batches()):
            #print(train_batch)
            optim.zero_grad()
#             if not multi_task:
#                 tags = torch.tensor(dev_batch['tags']).to(device)
#                 ABtags = torch.tensor(dev_batch['ABtags']).to(device)
#                 inputs = dev_batch['prompt_inputs']

#                 input_ids = torch.tensor(inputs['input_ids']).to(device)
#                 token_type_ids = torch.tensor(inputs['token_type_ids']).to(device)
#                 attention_mask = torch.tensor(inputs['attention_mask']).to(device)

#                 prompt_mask = torch.tensor(dev_batch['prompt_mask']).to(device)

#                 ABlogits, _ = model(input_ids, token_type_ids, attention_mask, ABtags, prompt_mask)
#                 loss = cross_entropy_mask(ABlogits, ABtags, prompt_mask)
#                 dev_losses.append(loss.cpu().detach().numpy())
#             else:
            multitask_ABtags = torch.tensor(dev_batch['multitask_ABtags']).to(device)
            multitask_Auxtags = torch.tensor(dev_batch['multitask_Auxtags']).to(device)

            inputs = dev_batch['multitask_prompt_inputs']
            input_ids = torch.tensor(inputs['input_ids']).to(device)
            token_type_ids = torch.tensor(inputs['token_type_ids']).to(device)
            attention_mask = torch.tensor(inputs['attention_mask']).to(device)

            multitask_prompt_mask = torch.tensor(dev_batch['multitask_prompt_mask']).to(device)
            auxiliary_prompt_mask = torch.tensor(dev_batch['auxiliary_prompt_mask']).to(device)

            ABlogits, Auxlogits = model(input_ids, token_type_ids, attention_mask)
            loss = cross_entropy_mask(ABlogits, multitask_ABtags, multitask_prompt_mask)
            #losses.append(loss.detach().cpu().numpy()) 

            aux_loss = cross_entropy_mask(Auxlogits, multitask_Auxtags, auxiliary_prompt_mask)                                                                            
            loss = aux_loss

            dev_losses.append(loss.cpu().detach().numpy())
        for test_batch in tqdm.tqdm(test_batches.get_batches()):
            #print(train_batch)
            optim.zero_grad()

#             if not multi_task:
#                 tags = torch.tensor(test_batch['tags']).to(device)
#                 ABtags = torch.tensor(test_batch['ABtags']).to(device)
#                 inputs = test_batch['prompt_inputs']

#                 input_ids = torch.tensor(inputs['input_ids']).to(device)
#                 token_type_ids = torch.tensor(inputs['token_type_ids']).to(device)
#                 attention_mask = torch.tensor(inputs['attention_mask']).to(device)

#                 prompt_mask = torch.tensor(test_batch['prompt_mask']).to(device)

#                 ABlogits, _ = model(input_ids, token_type_ids, attention_mask, ABtags, prompt_mask)
#                 loss = cross_entropy_mask(ABlogits, ABtags, prompt_mask)
#                 test_losses.append(loss.cpu().detach().numpy())
#             else:
            ABtags = torch.tensor(test_batch['multitask_ABtags']).to(device)
            multitask_Auxtags = torch.tensor(test_batch['multitask_Auxtags']).to(device)

            inputs = test_batch['multitask_prompt_inputs']
            input_ids = torch.tensor(inputs['input_ids']).to(device)
            token_type_ids = torch.tensor(inputs['token_type_ids']).to(device)
            attention_mask = torch.tensor(inputs['attention_mask']).to(device)

            prompt_mask = torch.tensor(test_batch['multitask_prompt_mask']).to(device)
            auxiliary_prompt_mask = torch.tensor(test_batch['auxiliary_prompt_mask']).to(device)

            ABlogits, Auxlogits = model(input_ids, token_type_ids, attention_mask)
            loss = cross_entropy_mask(ABlogits, ABtags, prompt_mask)
            #losses.append(loss.detach().cpu().numpy()) 

            aux_loss = cross_entropy_mask(Auxlogits, multitask_Auxtags, auxiliary_prompt_mask)                                                                            
            loss = aux_loss

            test_losses.append(loss.cpu().detach().numpy())

            ABprobs = nn.Softmax(dim=-1)(ABlogits) 

            ABprobs = ABprobs.cpu().detach().numpy()
            prompt_mask = prompt_mask.cpu().detach().numpy()
            ABtags = ABtags.cpu().detach().numpy()
            for ABprob, masks, tags in zip(ABprobs, prompt_mask, ABtags):
                masked_index = masks.nonzero()[-1]
                test_probs.append(ABprob[masked_index])
                test_tags.append(tags[masked_index])
            #if multi_task:
            Auxprobs = nn.Softmax(dim=-1)(Auxlogits)
            Auxprobs = Auxprobs.cpu().detach().numpy()
            auxiliary_prompt_mask = auxiliary_prompt_mask.cpu().detach().numpy()
            multitask_Auxtags = multitask_Auxtags.cpu().detach().numpy()
            for probs, masks, tags in zip(Auxprobs, auxiliary_prompt_mask, multitask_Auxtags):
                masked_indexes = masks.nonzero()[0]

                temp_probs = probs[masked_indexes][:,1]
                temp_tags = tags[masked_indexes]
#                 for masked_index in masked_indexes:
#                     prob = probs[masked_index]
#                     tag = tags[masked_index]

#                     temp_probs.append(prob[:,1]);
#                     temp_tags.append(tag)
                test_aux_probs.append(temp_probs)
                test_aux_tags.append(temp_tags)
        print("%.2f, %.2f"%(np.mean(dev_losses), np.mean(test_losses)))


    # In[51]:


    predict = [np.argmax(probs) for  probs in test_probs] #np.where(np.array(test_probs)>0.5, 1.0, 0.0)
    tags = np.array(test_tags)


    # In[52]:


    # list(zip(predict, tags))


    # In[53]:


    masked_index


    # In[54]:


    index = 0
    #tag = 'ABERRANT_BEHAVIOR'
    #tag = 'MED CHANGE'
    for ts, ps, c in zip(tags, predict, test_context):
        #print(ts[tag2id['MED CHANGE']], ps[tag2id['MED CHANGE']])
        if ts[ABtag2id[tag]] != ps[ABtag2id[tag]]:
            print(c, ts[ABtag2id[tag]], ps[ABtag2id[tag]])
        index += 1


    # In[55]:


    from sklearn.metrics import f1_score, precision_recall_curve, confusion_matrix
    from sklearn.metrics import precision_recall_fscore_support as score
    from sklearn import metrics


    # In[56]:


    precision, recall, fscore, support = score(tags, predict)

    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))


    # In[57]:


    macro_f1 = f1_score(tags, predict, average='macro')
    micro_f1 = f1_score(tags, predict, average='micro')
    f1s = f1_score(tags, predict, average=None)


    # In[58]:


    "%.2f %.2f"%(macro_f1*100, micro_f1*100)


    # In[59]:


    for id_key, f1 in  enumerate(f1s):
        print("%s: %.2f"%(id2ABtag[id_key], f1*100)) 


    # In[60]:


    conf_mat = confusion_matrix(tags, predict)

    print("\t"+"\t".join([tag[:7] for tag in ABtag2id.keys()]))
    for index_x in range(len(conf_mat)):
        true = conf_mat[index_x]

        label = "%7s"%id2ABtag[index_x][:7]+'\t'
        for index_y in range(len(conf_mat[index_x])):
            label += str(conf_mat[index_x][index_y]) + '\t'
        print(label)


    # In[61]:

    key = 0
    for prob, tag in zip(np.array(test_aux_probs).T, np.array(test_aux_tags).T):
        precision, recall, thresholds = precision_recall_curve(tag, prob)
        auprc = metrics.average_precision_score(tag, prob)
        print("%s: %.2f"%(id2tag[key], auprc*100)) 
        key += 1
    # key = 0
    # for prob, tag in zip(np.array(test_probs).T, tags.T):
    #     precision, recall, thresholds = precision_recall_curve(tag, prob)
    #     auprc = metrics.average_precision_score(tag, prob)
    #     print("%s: %.2f"%(id2tag[key], auprc*100)) 
    #     key += 1


    # In[ ]:





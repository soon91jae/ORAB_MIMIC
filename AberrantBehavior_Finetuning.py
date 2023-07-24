#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] ="1"

import random 
random.seed(0)

import collections
from collections import Counter

from itertools import islice, chain



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



device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[5]:


torch.cuda.current_device()


# In[6]:


import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoConfig


# In[7]:




# In[8]:


#MODEL_PATH = "emilyalsentzer/Bio_ClinicalBERT"
MODEL_PATH = "dmis-lab/biobert-v1.1"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


# In[9]:


bz = 8
epochs = 3
lr = 3e-5
folds = 5
multi_task = True
# In[10]:

print("bz:%d, epochs:%d, lr: %e, Multitask; %s"%(bz,epochs, lr, multi_task))

#data_path = '/home/vhabedkwons/Projects/Abberant_Behavior/dataset/extract_ehost_MIMIC/extract_ehost_UMass/extract_ehost/'
data_path = '/home/vhabedkwons/Projects/Abberant_Behavior/dataset/extract_ehost/'

data = pickle.load(open(data_path+'pkls.pkl','rb'))
raw_text_dict = pickle.load(open(data_path+'raw_text_dict.pkl','rb'))


data_dict = {}
sent_dict= {}

tag_cnt = Counter()
tag2id = {}
id2tag = {}
ABtag2id = {}
id2ABtag = {}


# In[11]:


temp_sent_id = 0
for example in data:
    start, end, instance, tag, context, doc_file = example
    
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
    
    if tag not in tag2id:# and 'ABERRANT_BEHAVIOR' not in tag
        if not multi_task and 'ABERRANT_BEHAVIOR' not in tag: continue 
        tag2id[tag] = len(tag2id)
    if tag not in ABtag2id and 'ABERRANT_BEHAVIOR' in tag: 
        ABtag2id[tag] = len(ABtag2id)
    tag_cnt[tag] += 1
for tag in tag2id.keys(): 
    id2tag[tag2id[tag]] = tag

ABtag2id['none'] = len(ABtag2id)
for tag in ABtag2id.keys():
    id2ABtag[ABtag2id[tag]] = tag


# In[12]:


# for key in data_dict.keys():
#     tags = []
#     for instance in data_dict[key]['instances']:
#         tags.append(instance['tag'])
#     if len(set(tags))>1:
#         print(data_dict[key])
# #data_dict[11]


# In[ ]:





# In[13]:


id2ABtag


# In[14]:


#data_dict[key]


# In[15]:


# print(len(data_dict))

# instance_num = 0
# for key in data_dict.keys():
#     instance_num += len(data_dict[key]['instances'])
# instance_num


# In[16]:


sent_keys = list(data_dict.keys())
random.shuffle(sent_keys)

kf = KFold(n_splits = folds)
kf.get_n_splits(sent_keys)

for i, (train_index, test_index) in enumerate(kf.split(sent_keys)):
    
    train_keys = [sent_keys[index] for index in train_index]
    #dev_keys = sent_keys[int(len(sent_keys) * 0.8):int(len(sent_keys) * 0.9)]
    test_keys = [sent_keys[index] for index in test_index]

    train_data = [data_dict[key] for key in train_keys]
    dev_data = [data_dict[key] for key in test_keys]#[data_dict[key] for key in dev_keys]
    test_data = [data_dict[key] for key in test_keys]


    # In[17]:


    #train_data[0]


    # In[18]:


    tag_cnt


    # In[19]:


    class data_processor(object):
        def __init__(self, tag2id, ABtag2id, tokenizer):
            self.tag2id = tag2id
            self.ABtag2id = ABtag2id
            self.tokenizer = tokenizer
            return


        def processing(self, data):
            tag2id = self.tag2id
            ABtag2id = self.ABtag2id
            tokenizer = self.tokenizer

            processed_data = []


            for example in data:


                context = example['context']; 
                instances = example['instances']
                input_ids = tokenizer.encode(context)


                tags = [0.0] * len(tag2id)
                for instance in instances:
                    tag = instance['tag']
                    if tag in tag2id:
                        tags[tag2id[tag]] = 1.0

                ABtags = [0] * 3
                for instance in instances:
                    tag = instance['tag']
                    if tag in ABtag2id:
                        ABtags[ABtag2id[tag]] = 1
                if sum(ABtags) == 0: ABtags[ABtag2id['none']] = 1
                elif sum(ABtags) == 2: ABtags[ABtag2id['SUGGEST_ABERRANT_BEHAVIOR']] = 0
                ABtags = np.argmax(ABtags)


                example_dict = {"context": context,
                                "input_ids": input_ids,
                                "tags": tags,
                                "ABtags": ABtags}
                processed_data.append(example_dict)

            return processed_data


    # In[20]:


    processor = data_processor(tag2id, ABtag2id, tokenizer)

    train_inputs = processor.processing(train_data)
    dev_inputs = processor.processing(dev_data)
    test_inputs = processor.processing(test_data)


    # In[21]:


    #train_inputs[0]


    # In[22]:


    tag2id, ABtag2id


    # In[23]:


    # import pandas as pd

    # data = {tag:[] for tag in tag2id.keys()}
    # for train_input in train_inputs:
    #     tags = train_input['tags']

    #     for tag_id, tag in enumerate(tags):

    #         data[id2tag[tag_id]].append(tag)
    # df = pd.DataFrame(data)


    # In[24]:


    # from scipy.stats import chi2_contingency

    # def cramers_V(var1, var2):
    #     crosstab = np.array(pd.crosstab(var1, var2))
    #     stat = chi2_contingency(crosstab)[0]
    #     obs = np.sum(crosstab)
    #     mini = min(crosstab.shape)-1
    #     return (stat/(obs*mini))


    # In[25]:


    # import itertools

    # chi2dict = {}
    # for col1, col2 in itertools.permutations(df.columns,2):
    #     print(col1, col2, cramers_V(df[col1], df[col2]))


    # In[26]:


    # for test_input in test_inputs:
    #     context = test_input['context']
    #     tags = test_input['tags']

    #     if tags[tag2id['MED CHANGE']] == 1.0:
    #         print(context)


    # In[27]:


    class batchfier(object):
        def __init__(self, inputs, tokenizer, bz=32, shuffle=True):
            self.inputs =  inputs#inputs
            self.bz = bz
            self.shuffle = shuffle
            self.tokenizer = tokenizer

        def get_batches(self):
            inputs= self.inputs
            bz = self.bz
            shuffle = self.shuffle
            tokenizer = self.tokenizer

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
                tags = [inputs[index]['tags'] for index in batch_index]
                ABtags = [inputs[index]['ABtags'] for index in batch_index]
                contexts = [inputs[index]['context'] for index in batch_index]
                input_ids = tokenizer.batch_encode_plus(contexts,  pad_to_max_length=True)



                batch_input = {'tags': tags,
                               'ABtags': ABtags,
                               'contexts': contexts,
                               'inputs': input_ids}
                batch_inputs.append(batch_input)
            return batch_inputs   


    # In[28]:


    train_batches = batchfier(train_inputs, tokenizer, bz)
    test_batches = batchfier(test_inputs, tokenizer, bz, shuffle=False)
    dev_batches = batchfier(dev_inputs, tokenizer, bz, shuffle=False)


    # In[29]:


    class BaselineModel(torch.nn.Module):
        def __init__(self, MODEL_PATH, tag2id, ABtag2id):
            super(BaselineModel, self).__init__()

            self.plm = AutoModel.from_pretrained(MODEL_PATH)
            self.config = AutoConfig.from_pretrained(MODEL_PATH)
            self.dropout = torch.nn.Dropout(0.3)
            self.classifier = torch.nn.Linear(self.config.hidden_size, len(tag2id))
            self.ABclassifier = torch.nn.Linear(self.config.hidden_size, len(ABtag2id))
        def forward(self, input_ids, token_type_ids, attention_mask):
            _, hiddens = self.plm(input_ids, token_type_ids, attention_mask)
            outputs = self.dropout(hiddens)
            logits = self.classifier(outputs)
            ABlogits = self.ABclassifier(outputs)

            return logits, ABlogits, hiddens


    # In[30]:


    model = BaselineModel(MODEL_PATH, tag2id, ABtag2id)
    model = model.to(device)


    # In[31]:


    # AutoConfig.from_pretrained(MODEL_PATH).hidden_size
    total_step = len(train_batches.get_batches())*epochs

    optim = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optim, 
                                                num_warmup_steps=total_step*0.1,
                                                num_training_steps=total_step)
    criterion = nn.CrossEntropyLoss()
    aux_criterion = nn.BCEWithLogitsLoss()


    # In[32]:


    losses = []
    best_model = None
    best_loss = None
    for epoch in tqdm.tqdm(range(epochs)):
        for train_batch in tqdm.tqdm(train_batches.get_batches()):
            #print(train_batch)
            optim.zero_grad()

            tags = torch.tensor(train_batch['tags']).to(device)
            ABtags = torch.tensor(train_batch['ABtags']).to(device)

            inputs = train_batch['inputs']
            input_ids = torch.tensor(inputs['input_ids']).to(device)
            token_type_ids = torch.tensor(inputs['token_type_ids']).to(device)
            attention_mask = torch.tensor(inputs['attention_mask']).to(device)

            logits, ABlogits, hiddens = model(input_ids, token_type_ids, attention_mask)
            aux_loss = aux_criterion(logits, tags)
            #print(ABtags)
            #loss = criterion(ABlogits, ABtags)
            loss = aux_loss




            losses.append(loss.cpu().detach().numpy())

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

                tags = torch.tensor(dev_batch['tags']).to(device)
                ABtags = torch.tensor(dev_batch['ABtags']).to(device)

                inputs = dev_batch['inputs']
                input_ids = torch.tensor(inputs['input_ids']).to(device)
                token_type_ids = torch.tensor(inputs['token_type_ids']).to(device)
                attention_mask = torch.tensor(inputs['attention_mask']).to(device)

                logits, ABlogits, hiddens = model(input_ids, token_type_ids, attention_mask)
                aux_loss = aux_criterion(logits, tags)
                loss = criterion(ABlogits, ABtags)

                dev_losses.append(loss.cpu().detach().numpy())
            for test_batch in test_batches.get_batches():
                #print(train_batch)
                optim.zero_grad()

                tags = torch.tensor(test_batch['tags']).to(device)
                ABtags = torch.tensor(test_batch['ABtags']).to(device)

                inputs = test_batch['inputs']
                input_ids = torch.tensor(inputs['input_ids']).to(device)
                token_type_ids = torch.tensor(inputs['token_type_ids']).to(device)
                attention_mask = torch.tensor(inputs['attention_mask']).to(device)

                logits, ABlogits, hiddens = model(input_ids, token_type_ids, attention_mask)
                aux_loss = aux_criterion(logits, tags)
                loss = criterion(ABlogits, ABtags)

                test_losses.append(loss.cpu().detach().numpy())
            print("%.2f, %.2f"%(np.mean(dev_losses), np.mean(test_losses)))




    # In[33]:


    ABlogits.shape


    # In[34]:


    ABtags


    # In[35]:


    import matplotlib.pyplot as plt

    xpoints = list(range(len(losses)))
    plt.plot(xpoints, losses)
    plt.show()


    # In[ ]:





    # In[36]:


    test_losses=[]
    dev_losses = []

    test_probs = []
    test_tags = []
    test_aux_probs = []
    test_aux_tags = []
    test_contextss = []

    model.eval()
    with torch.no_grad():
        for dev_batch in tqdm.tqdm(dev_batches.get_batches()):
            #print(train_batch)
            optim.zero_grad()

            tags = torch.tensor(dev_batch['tags']).to(device)
            ABtags = torch.tensor(dev_batch['ABtags']).to(device)

            inputs = dev_batch['inputs']
            input_ids = torch.tensor(inputs['input_ids']).to(device)
            token_type_ids = torch.tensor(inputs['token_type_ids']).to(device)
            attention_mask = torch.tensor(inputs['attention_mask']).to(device)

            logits, ABlogits, hiddens = model(input_ids, token_type_ids, attention_mask)
            aux_loss = aux_criterion(logits, tags)
            loss = criterion(ABlogits, ABtags)

            dev_losses.append(loss.cpu().detach().numpy())
        for test_batch in tqdm.tqdm(test_batches.get_batches()):
            #print(train_batch)
            optim.zero_grad()
            test_contexts = test_batch['contexts']
            tags = torch.tensor(test_batch['tags']).to(device)
            ABtags = torch.tensor(test_batch['ABtags']).to(device)

            inputs = test_batch['inputs']
            input_ids = torch.tensor(inputs['input_ids']).to(device)
            token_type_ids = torch.tensor(inputs['token_type_ids']).to(device)
            attention_mask = torch.tensor(inputs['attention_mask']).to(device)

            logits, ABlogits, hiddens = model(input_ids, token_type_ids, attention_mask)
            aux_loss = aux_criterion(logits, tags)
            loss = criterion(ABlogits, ABtags)

            test_losses.append(loss.cpu().detach().numpy())
            for tag in test_batch['ABtags']:
                test_tags.append(tag)
            ABprobs = nn.Softmax(dim=-1)(ABlogits) 
            for ABprob in ABprobs.cpu().detach().numpy():
                test_probs.append(ABprob)
            for test_context in test_contexts:
                test_contextss.append(test_context)
                
            for tag in test_batch['tags']:
                test_aux_tags.append(tag)
            probs = torch.sigmoid(logits)
            for prob in probs.cpu().detach().numpy():
                test_aux_probs.append(prob)
        print("%.2f, %.2f"%(np.mean(dev_losses), np.mean(test_losses)))


    # In[37]:


    predict = [np.argmax(probs) for  probs in test_probs] #np.where(np.array(test_probs)>0.5, 1.0, 0.0)
    tags = np.array(test_tags)


    # In[38]:


    ABtag2id


    # In[ ]:





    # In[40]:


    index = 0
    #tag = 'ABERRANT_BEHAVIOR'
    #tag = 'MED CHANGE'
    error_cases = {'context': [], 'predict': [], 'tag': []}
    for ts, ps, c in zip(tags, predict, test_contextss):
        #print(ts[tag2id['MED CHANGE']], ps[tag2id['MED CHANGE']])
        if ts != ps:
            #print("\t".join([c, id2ABtag[ts], id2ABtag[ps]]))
            error_cases['context'].append(c)
            error_cases['predict'].append(id2ABtag[ps])
            error_cases['tag'].append(id2ABtag[ts])
        index += 1

    import pandas as pd 
    pd.DataFrame(error_cases).to_csv('error.csv')


    # In[41]:


    from sklearn.metrics import f1_score, precision_recall_curve, confusion_matrix
    from sklearn.metrics import precision_recall_fscore_support as score
    from sklearn import metrics


    # In[42]:


    precision, recall, fscore, support = score(tags, predict)

    print('precision: {}'.format(precision))
    print('recall: {}'.format(recall))
    print('fscore: {}'.format(fscore))
    print('support: {}'.format(support))


    # In[43]:


    macro_f1 = f1_score(tags, predict, average='macro')
    micro_f1 = f1_score(tags, predict, average='micro')
    f1s = f1_score(tags, predict, average=None)


    # In[44]:


    "Macro F1: %.2f Micro F1: %.2f"%(macro_f1*100, micro_f1*100)


    # In[45]:


    for id_key, f1 in  enumerate(f1s):
        print("%s: %.2f"%(id2ABtag[id_key], f1*100)) 


    # In[46]:


    conf_mat = confusion_matrix(tags, predict)

    print("\t"+"\t".join([tag[:7] for tag in ABtag2id.keys()]))
    for index_x in range(len(conf_mat)):
        true = conf_mat[index_x]

        label = "%7s"%id2ABtag[index_x][:7]+'\t'
        for index_y in range(len(conf_mat[index_x])):
            label += str(conf_mat[index_x][index_y]) + '\t'
        print(label)


# In[ ]:

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





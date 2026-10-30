#@contact    Sejoon Oh (soh337@gatech.edu), Georgia Institute of Technology
#@version    1.0
#@date       2022-08-04
#Rank List Sensitivity of Recommender Systems to Interaction Perturbations
#This software is free of charge under research purposes.
#For commercial purposes, please contact the main author.
import numpy as np
import random
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #Make cuDNN/cuBLAS pick deterministic kernels so that two runs on identical data give identical models
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def frbo(list1, list2, k=10, p=0.9):
    #Finite Rank-Biased Overlap@k (Betello et al., ECIR 2024): RBO truncated at depth k and divided by its maximum (1-p^k), so identical lists score 1
    score = sum(p**(d-1)*len(set(list1[:d])&set(list2[:d]))/d for d in range(1,k+1))
    return (1-p)*score/(1-p**k)

def data_cleaning(raw_data,test_data_ratio,min_interactions=10):
    #Remove users with fewer than min_interactions interactions (Section 3.1); the original code computed this filter but never applied it
    (users,counts) = np.unique(raw_data[:,0],return_counts = True)
    raw_data = raw_data[np.isin(raw_data[:,0],users[counts>=min_interactions])]

    #Map users and items to contiguous indices after filtering, since item indices are used as model output positions
    unique_users = sorted(list(set(raw_data[:, 0])))
    unique_items = sorted(list(set(raw_data[:, 1])))
    user_dic = {user:idx for (idx,user) in enumerate(unique_users)}
    item_dic = {item:idx for (idx,item) in enumerate(unique_items)}

    sequence_dic =  {idx:[] for idx in range(len(unique_users))}
    new_data = []
    for row in raw_data:
        new_data.append([user_dic[row[0]],item_dic[row[1]],row[2],0])

    new_data = np.array(new_data)

    for i in range(new_data.shape[0]):
        sequence_dic[int(new_data[i,0])].append([i,int(new_data[i,1]),new_data[i,2]])
    
    for user in sequence_dic.keys():
        cur_test = int(test_data_ratio*len(sequence_dic[user]))
        for i in range(cur_test):
            interaction = sequence_dic[user].pop()
            new_data[interaction[0],3] = 1

    return new_data

def train_test_generator(data, look_back):

    train,test = [],[]
    unique_users = set(data[:,0])
    items_per_user = {int(user):[0 for i in range(look_back)] for user in unique_users}
    
    for (idx,row) in enumerate(data):
        user,item,time = int(row[0]),int(row[1]),row[2]
        items_per_user[user] = items_per_user[user][1:]+[item+1]
        current_items = items_per_user[user]
        if row[3]==0:
            train.append([current_items[:-1],current_items[-1]])                                                                                            
        else:
            test.append([current_items[:-1],current_items[-1]])
                                                                
    return train,test



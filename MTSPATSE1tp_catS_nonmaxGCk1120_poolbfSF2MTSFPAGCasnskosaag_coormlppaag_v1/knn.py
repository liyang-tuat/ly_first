# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 14:34:40 2019

@author: 81906
"""
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from time import time
import numpy as np
from torch.autograd import Variable
import scipy
from scipy.spatial import cKDTree
from scipy.sparse.linalg import eigsh
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def adjacency(dist,idx,attention_f):
    """Return the adjacency matrix of a kNN graph."""
    """
    图的每一条边都有权重，计算由边的权重组成的邻接矩阵W
    论文中的式（1）
    """
    M,k = dist.shape
    assert M, k == idx.shape
    assert dist.min() >= 0
    
    #weight matrix
    #I=[0,1,2,...,M]然后将每一个数都复制k次，I=[1,1,1,...1,2,2,2,...,2,...,M,M,M,...,M]
    #I的size是M*k
    I = np.arange(0,M).repeat(k)
    J = idx.reshape(M*k)
   
    W1 = torch.zeros(M,M)
    if attention_f == 1:   
        sigma2 = np.mean(dist[:,-1]) ** 2
        dist = np.exp(-dist**2/sigma2)
        V = dist.reshape(M*k)   
        for i in range(len(I)):
            W1[I[i],J[i]] = torch.tensor(V[i])
    else:
        for i in range(len(I)):
            W1[I[i],J[i]] = 1.0
             
    W1 = W1 - torch.eye(M)
    """
    bigger = W1.t() > W1
    W1 = W1 - W1.mul(bigger.float()) + W1.t().mul(bigger.float())
    #print("W1 is:",W1)
    """
    return W1

def adjacency1(dist,idx):
    """Return the adjacency matrix of a kNN graph."""
    """
    图的每一条边都有权重，计算由边的权重组成的邻接矩阵W
    论文中的式（1）
    """
    M,k = dist.shape
    assert M, k == idx.shape
    assert dist.min() >= 0
    
    #weight matrix
    #I=[0,1,2,...,M]然后将每一个数都复制k次，I=[1,1,1,...1,2,2,2,...,2,...,M,M,M,...,M]
    #I的size是M*k
    I = np.arange(0,M).repeat(k)
    J = idx.reshape(M*k)
   
    W1 = torch.zeros(M,M)
    W1_one = torch.zeros(M,M) 
    
    sigma2 = np.mean(dist[:,-1]) ** 2
    dist = np.exp(-dist**2/sigma2)
    V = dist.reshape(M*k)
    for i in range(len(I)):
        W1[I[i],J[i]] = torch.tensor(V[i])
        W1_one[I[i],J[i]] = 1.0
             
    W1 = W1 - torch.eye(M)
    W1_one = W1_one - torch.eye(M)
   
    return W1,W1_one

def knn(x,k,attention_f):
    """
    input:
        x:一个batch,里面有多个3D模型点云，[B,C,N]
        k:近旁点个数
    output:
        idx:近旁点的索引
    """
    device = torch.device("cuda")
    x = x.transpose(2,1).cpu().detach().numpy()#[B,N,C]
    batch_size,num_points,num_dims = x.shape
   
    w = torch.zeros(batch_size,num_points,num_points)
    #w_one = torch.zeros(batch_size,num_points,num_points)
    
    #ii = torch.LongTensor(ii)
    #idx = ii
    for i in range(batch_size):
        one_model = x[i]
        #print("one_model size is:",one_model.shape)
        tree = cKDTree(data = one_model)
        dd,ii = tree.query(one_model,k=k)
        
        w[i] = adjacency(dd,ii,attention_f).to(device)
        #w[i],w_one[i] = adjacency1(dd,ii)
        #ii = torch.LongTensor(ii)
        #idx = torch.cat((idx,ii),dim=0)
    #idx = idx.view(batch_size,num_points,-1)#[B,N,k]
    
    w = w.view(batch_size,num_points,-1)
    #w_one = w_one.view(batch_size,num_points,-1).to(device)
     
    return w

def knn1(x,k):
    """
    input:
        x:一个batch,里面有多个3D模型点云，[B,C,N]
        k:近旁点个数
    output:
        idx:近旁点的索引
    """
    
    inner = -2*torch.matmul(x.transpose(2,1),x)#[B,N,N]
    xx = torch.sum(x**2,dim=1,keepdim=True)#[B,1,N]
    #计算点与点之间的距离，并且将自己与自己的距离置0，与其他点的距离全部变成负数（便于找近旁点）
    pairwise_distance = -xx - inner - xx.transpose(2,1) 
    #找近旁点索引，最大的k个置
    idx = pairwise_distance.topk(k=k,dim=-1)[1]#[B,N,k]
    #print("idx is:",idx)
    
    return idx

def get_graph_feature(x,k=20,idx=None):
    """
    计算特征，论文Dynamic Graph CNN for Learning on Point Clouds中式（7）
    input:
        x:一个batch,里面有多个3D模型点云，[B,C,N]
        k:近旁点个数
        idx:点的索引
    output:
        feature:特征
    """
    device = torch.device("cuda")
    #device = torch.device("cpu")
    
    batch_size = x.shape[0]
    num_dims = x.shape[1]
    num_points = x.shape[2]
    if idx is None:
        idx = knn1(x,k)#[B,N,k]
    
    #用于在各个模型合并成一个的矩阵中找点
    idx_base = torch.arange(0,batch_size,device=device).view(-1,1,1)*num_points
    #idx_base = torch.arange(0,batch_size).view(-1,1,1)*num_points
    idx = idx + idx_base
    idx = idx[:,:,1:k].contiguous()
    idx = idx.view(-1)
    #print("idx is:",idx)
    
    #如果我们在 transpose、permute 操作后执行 view会报错，需要加上.contiguous()
    x = x.transpose(2,1).contiguous()#[B,N,C]
    
    #找出每个点的k个最近邻点
    feature = x.view(batch_size*num_points,-1)[idx,:]
    feature = feature.view(batch_size,num_points,k-1,num_dims)
    #在第三个维度上复制k次
    x = x.view(batch_size,num_points,1,num_dims)
    feature = feature-x
    #print("diff feature is:",feature)
    diff = feature
    f_abs = torch.abs(feature)
    #print("distent is:",f_abs)
    #feature = f_abs

    feature = f_abs
            
    return feature,diff

def get_graph_feature_A(x,xyz,k=20,idx=None):
    """
    计算特征，论文Dynamic Graph CNN for Learning on Point Clouds中式（7）
    input:
        x:一个batch,里面有多个3D模型点云，[B,C,N]
        k:近旁点个数
        idx:点的索引
    output:
        feature:特征
    """
    device = torch.device("cuda")
    #device = torch.device("cpu")
    
    batch_size = x.shape[0]
    num_dims = x.shape[1]
    num_points = x.shape[2]
    if idx is None:
        idx = knn1(xyz,k)#[B,N,k]
    
    #用于在各个模型合并成一个的矩阵中找点
    idx_base = torch.arange(0,batch_size,device=device).view(-1,1,1)*num_points
    #idx_base = torch.arange(0,batch_size).view(-1,1,1)*num_points
    idx = idx + idx_base
    idx = idx[:,:,:k].contiguous()
    idx = idx.view(-1)
    #print("idx is:",idx)
    
    #如果我们在 transpose、permute 操作后执行 view会报错，需要加上.contiguous()
    x = x.transpose(2,1).contiguous()#[B,N,C]
    xyz = xyz.transpose(2,1).contiguous()#[B,N,C]
    
    #找出每个点的k个最近邻点
    feature = x.view(batch_size*num_points,-1)[idx,:]
    feature = feature.view(batch_size,num_points,k,num_dims)
    n_xyz = xyz.view(batch_size*num_points,-1)[idx,:]
    n_xyz = n_xyz.view(batch_size,num_points,k,-1)
    
    xyz = xyz.view(batch_size,num_points,1,-1)
    p = n_xyz - xyz
    
    diff = p
    return feature,p,diff
        
"""  
batch1 = torch.tensor([[[1,2,3],[4,1,3],[7,2,9],[17,11,2],[1,1,1]],[[9,8,0],[0,2,5],[10,2,3],[0,0,1],[2,2,0]]],dtype=torch.float)
batch1 = batch1.transpose(2,1)#[B,C,N]
print(batch1.shape)
node,features,diff,node_h,QK,node_v,p = get_graph_feature(batch1,k=4,flage=1,mean=True,q=batch1,ky=batch1,v=batch1)#[B,N,k,C] 
print(features)
#print("node is:",node)
print("QK is:",QK)
print("node_v is:",node_v)
print("p is:",p)

t1 = node.repeat(1,1,1,3).view(2,5,-1,3)
#print("t1 is:",t1)
t2 = node.repeat(1,1,3,1)
#print("t2 is:",t2.shape)
t3 = torch.abs(t2-t1).view(2,5,3,3,3)
#print("t3 is:",t3)
t3 = torch.sum(t3,dim=3)
print("t3 is:",t3)
print("t3 is:",t3.shape)
A = torch.bmm(batch1.transpose(2,1),batch1)
print("A is:",A)
print("a is:",a)
zero_vec = -9e-15*torch.ones_like(A)
attention = torch.where(a>0,A,zero_vec)
print("attention is:",attention)
attention = F.softmax(attention,dim=2)
print("attention is:",attention)
t = torch.sum(attention,dim=2)
print(t)
at = attention[idx[:,:,0],idx[:,:,1],idx[:,:,2]].view(2,5,4,1)#[B,N,k,1]
print("at is:",at)
n_at = at[:,:,1:,:]
print(n_at)
s_at = at[:,:,0,:]
print(s_at.shape)

inner = -2*torch.matmul(batch1.transpose(2,1),batch1)#[B,N,N]
xx = torch.sum(batch1**2,dim=1,keepdim=True)#[B,1,N]
#计算点与点之间的距离，并且将自己与自己的距离置0，与其他点的距离全部变成负数（便于找近旁点）
pairwise_distance = xx + inner + xx.transpose(2,1)
print("d is:",pairwise_distance)
dis = torch.sqrt(pairwise_distance)
print("dis is:",dis)

sigma2 = torch.mean(pairwise_distance,dim=2,keepdim=True)
print("sigma2 is:",sigma2)
w = torch.exp(-pairwise_distance/sigma2)
print("w is:",w)

t = torch.bmm(batch1.transpose(2,1),batch1)
print("t is:",t)
m = F.sigmoid(t)
print("m is:",m)

e_p = m[idx[:,:,0],idx[:,:,1],idx[:,:,2]].view(2,5,2,1)
print("e_p is:",e_p)

w = torch.FloatTensor([1,0,1,0,1,1]).view(1,6)
print("w is:",w.shape)
features = features.permute(0,3,1,2)
features = features.view(2,6,-1)
print("features is:",features)
r = w.matmul(features).view(2,1,5,2)
print("r is:",r)

att = torch.ones(3,1)
H = batch1.transpose(2,1).matmul(att)
print("att is:",att)
print("H is:",H)
H_att = F.softmax(H,dim=1)
print("H_att is:",H_att)
features = get_graph_feature(batch1,k=3,flage=1,score=H_att) 
print(features)

pool_idx = torch.sort(H_att,dim=1,descending=True)[1][:,0:4,:]
print("pool_idx is:",pool_idx)
device = torch.device("cpu")
idx_base = torch.arange(0,2,device=device).view(-1,1,1)*5
print("idx_base is:",idx_base)
pool_idx = (pool_idx.view(2,4,1) + idx_base).view(-1,1)
print("pool_idx is:",pool_idx)
points = batch1.transpose(2,1).view(2*5,-1)[pool_idx,:].view(2,4,-1).transpose(2,1)
print("points is:",points)
H_att1 = H_att.view(2*5,-1)[pool_idx,:].view(2,4,-1).transpose(2,1)
print("H_att1 is:",H_att1)
"""
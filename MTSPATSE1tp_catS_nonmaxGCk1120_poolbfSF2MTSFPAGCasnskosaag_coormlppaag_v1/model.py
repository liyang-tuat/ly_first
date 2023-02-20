# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 20:56:55 2019

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
from knn import knn,knn1,get_graph_feature,get_graph_feature_A
from pointnet2_utils import PointNetSetAbstraction,PointNetSetAbstractionMsg,PointNetFeaturePropagation
from ASNL_utils import pointASNL,pointASNLall

class Aggregator(nn.Module):
    def __init__(self,num_points,cuda=False,agg_flage=0):
        super(Aggregator,self).__init__()
        self.num_points = num_points
        self.cuda = cuda
        self.agg_flage = agg_flage
        """
        self.attention_w = nn.Parameter(torch.FloatTensor(self.batch_size,self.num_points,self.num_points))
        init.kaiming_normal_(self.attention_w)
        """  
        
    def forward(self,features,w,edge_flage):
        #batch_to_feats = torch.empty(batch_size,num_points,3).cuda()
        w = w.cuda()
        features = features.cuda()
        #a_w = w.mul(self.attention_w)
        batch_to_feats = w.matmul(features)
        
        if edge_flage == 1:
            num_neigh = w.sum(dim=2,keepdim=True)
            batch_to_feats = batch_to_feats - num_neigh * features
        
        batch_to_feats = batch_to_feats.permute(0,2,1).cuda()
        return batch_to_feats

class GraphSAGE_Pointnet(nn.Module):
    def __init__(self,num_points,k,pool_n,num_part):
        super(GraphSAGE_Pointnet,self).__init__()
        #self.batch_size = batch_size
        self.num_points = num_points
        self.k = k
        self.pool_n = pool_n
    
        self.device = torch.device("cuda")
        
        self.atwq01_1m = nn.Conv2d(3,3,1,bias=False)
        self.atwk01_1m = nn.Conv2d(3,3,1,bias=False)
        
        self.atwq01_1 = nn.Conv2d(3,64,1,bias=False)
        self.atwkv01_1 = nn.Conv2d(3,64*2,1,bias=False)
        self.atwa01_1 = nn.Conv2d(64,64,1,bias=False)
        self.bnatwa01_1 = nn.BatchNorm2d(64)
        self.atwa01_10 = nn.Conv2d(1,64,1,bias=False)
        self.atwa01_11 = nn.Conv2d(64,1,1,bias=False)
        self.atwp01_1 = nn.Conv2d(3,64,1,bias=False)
        self.bnatwp01_1 = nn.BatchNorm2d(64)
        self.atwp01_11 = nn.Conv2d(64,64,1,bias=False)
        self.bconv01_1 = nn.Conv2d(3,1,kernel_size=1,bias=False)
        self.bbn01_1 = nn.BatchNorm2d(1)
        self.bvconv01_1 = nn.Conv2d(3,3,kernel_size=1,bias=False)
        self.bvbn01_1 = nn.BatchNorm2d(3)
        self.bww01_1 = nn.Conv2d(3+1,1,1,bias=False)
        self.bwbn01_1 = nn.BatchNorm2d(1)
        self.wga01 = nn.Conv1d(3+2+3,3,1,bias=False)
        self.bnwga01 = nn.BatchNorm1d(3)
        self.conv01_1 = nn.Conv1d(64,64,kernel_size=1,bias=False)
        self.bn01_1c = nn.BatchNorm1d(64)
        self.sk_conv01_1 = nn.Conv1d(3*2,64,kernel_size=1,bias=False)
        self.sk_bn01_1 = nn.BatchNorm1d(64)
        self.convga01_1a = nn.Conv1d(64+64,64,kernel_size=1,bias=False)
        self.bnconvga01_1a = nn.BatchNorm1d(64)
        
        self.conv01_sc0 = nn.Conv1d(64+64,64,kernel_size=1,bias=False)
        self.bn01_sc0 = nn.BatchNorm1d(64)
        self.conv01_sc0_h = nn.Conv1d(64,64,kernel_size=1,bias=False)
        self.bn01_sc0_h = nn.BatchNorm1d(64)
        self.bn01_0 = nn.BatchNorm1d(64)
        
        self.co_conv01_2 = nn.Conv1d(3,32,kernel_size=1,bias=False)
        self.co_bn01_2 = nn.BatchNorm1d(32)
        
        self.atwq01_2m = nn.Conv2d(64,64,1,bias=False)
        self.atwk01_2m = nn.Conv2d(64,64,1,bias=False)
        
        self.atwq01_2 = nn.Conv2d(64,64,1,bias=False)
        self.atwkv01_2 = nn.Conv2d(64,64*2,1,bias=False)
        self.atwa01_2 = nn.Conv2d(64,64,1,bias=False)
        self.bnatwa01_2 = nn.BatchNorm2d(64)
        self.atwa01_20 = nn.Conv2d(1,64,1,bias=False)
        self.atwa01_21 = nn.Conv2d(64,1,1,bias=False)
        self.atwp01_2 = nn.Conv2d(32,64,1,bias=False)
        self.bnatwp01_2 = nn.BatchNorm2d(64)
        self.atwp01_21 = nn.Conv2d(64,64,1,bias=False)
        self.bconv01_2 = nn.Conv2d(32,1,kernel_size=1,bias=False)
        self.bbn01_2 = nn.BatchNorm2d(1)
        self.bvconv01_2 = nn.Conv2d(32,32,kernel_size=1,bias=False)
        self.bvbn01_2 = nn.BatchNorm2d(32)
        self.bww01_2 = nn.Conv2d(32+1,1,kernel_size=1,bias=False)
        self.bwbn01_2 = nn.BatchNorm2d(1)
        self.conv01_self = nn.Conv1d(32+2+64,64,kernel_size=1,bias=False)
        self.bnwga01_2 = nn.BatchNorm1d(64)
        self.conv01_2 = nn.Conv1d(64,64,kernel_size=1,bias=False)
        self.bn01_2c = nn.BatchNorm1d(64)
        self.sk_conv01_2 = nn.Conv1d(64+32,64,kernel_size=1,bias=False)
        self.sk_bn01_2 = nn.BatchNorm1d(64)
        self.convga01_2a = nn.Conv1d(64+64,64,kernel_size=1,bias=False)
        self.bnconvga01_2a = nn.BatchNorm1d(64)
        
        self.conv01_sc1 = nn.Conv1d(64+64,64,kernel_size=1,bias=False)
        self.bn01_sc1 = nn.BatchNorm1d(64)
        self.conv01_sc1_h = nn.Conv1d(64,64,kernel_size=1,bias=False)
        self.bn01_sc1_h = nn.BatchNorm1d(64)
        self.bn01_1 = nn.BatchNorm1d(64)
        
        self.wga01_line1 = nn.Conv1d(64*2,512,kernel_size=1,bias=False)
        self.bnga01_line1 = nn.BatchNorm1d(512)
        
        self.conv_d01a = nn.Conv1d(3,64,kernel_size=1,bias=False)
        self.bnconv_d01a = nn.BatchNorm1d(64)
        self.conv_d01b = nn.Conv1d(64,64,kernel_size=1,bias=False)
        self.bnconv_d01b = nn.BatchNorm1d(64)
        self.conv_d01c = nn.Conv1d(64,64,kernel_size=1,bias=False)
        self.bnconv_d01c = nn.BatchNorm1d(64)
        
        self.weight_d01 = nn.Parameter(torch.FloatTensor(64,64*3))
        init.kaiming_uniform_(self.weight_d01,a=0.2)
        self.bn_d01 = nn.BatchNorm1d(64)
        
        self.asnl01_1 = pointASNL(npoint=512,nsample=self.k[2],
                                             in_channel=64+3,b_c=32,b_c1=32,mlp_list=[32,64+1],l_mlp_list=[64,64],NL=False)
        
        self.co_conv01_4 = nn.Conv1d(3,32,kernel_size=1,bias=False)
        self.co_bn01_4 = nn.BatchNorm1d(32)
        
        self.atwq01_4m = nn.Conv2d(64,64,kernel_size=1,bias=False)
        self.atwk01_4m = nn.Conv2d(64,64,kernel_size=1,bias=False)
        
        self.atwq01_4 = nn.Conv2d(64,128,kernel_size=1,bias=False)
        self.atwkv01_4 = nn.Conv2d(64,128*2,kernel_size=1,bias=False)
        self.atwa01_4 = nn.Conv2d(128,128,kernel_size=1,bias=False)
        self.bnatwa01_4 = nn.BatchNorm2d(128)
        self.atwa01_40 = nn.Conv2d(1,128,kernel_size=1,bias=False)
        self.atwa01_41 = nn.Conv2d(128,1,kernel_size=1,bias=False)
        self.atwp01_4 = nn.Conv2d(32,128,kernel_size=1,bias=False)
        self.bnatwp01_4 = nn.BatchNorm2d(128)
        self.atwp01_41 = nn.Conv2d(128,128,kernel_size=1,bias=False)
        self.bconv01_4 = nn.Conv2d(32,1,kernel_size=1,bias=False)
        self.bbn01_4 = nn.BatchNorm2d(1)
        self.bvconv01_4 = nn.Conv2d(32,32,kernel_size=1,bias=False)
        self.bvbn01_4 = nn.BatchNorm2d(32)
        self.bww01_4 = nn.Conv2d(32+1,1,kernel_size=1,bias=False)
        self.bwbn01_4 = nn.BatchNorm2d(1)
        self.wga01_4 = nn.Conv1d(32+2+64,64,kernel_size=1,bias=False)
        self.bnwga01_4 = nn.BatchNorm1d(64)
        self.conv01_4 = nn.Conv1d(128,128,kernel_size=1,bias=False)
        self.bn01_4c = nn.BatchNorm1d(128)
        self.sk_conv01_4 = nn.Conv1d(64+32,128,kernel_size=1,bias=False)
        self.sk_bn01_4 = nn.BatchNorm1d(128)
        self.convga01_4a = nn.Conv1d(128+128,128,kernel_size=1,bias=False)
        self.bnconvga01_4a = nn.BatchNorm1d(128)
        
        self.conv01_sc3 = nn.Conv1d(128+128,128,kernel_size=1,bias=False)
        self.bn01_sc3 = nn.BatchNorm1d(128)
        self.conv01_sc3_h = nn.Conv1d(128,128,kernel_size=1,bias=False)
        self.bn01_sc3_h = nn.BatchNorm1d(128)
        self.bn01_3 = nn.BatchNorm1d(128)
        
        self.co_conv01_5 = nn.Conv1d(32,64,kernel_size=1,bias=False)
        self.co_bn01_5 = nn.BatchNorm1d(64)
        
        self.atwq01_5m = nn.Conv2d(128,128,kernel_size=1,bias=False)
        self.atwk01_5m = nn.Conv2d(128,128,kernel_size=1,bias=False)
        
        self.atwq01_5 = nn.Conv2d(128,128,kernel_size=1,bias=False)
        self.atwkv01_5 = nn.Conv2d(128,128*2,kernel_size=1,bias=False)
        self.atwa01_5 = nn.Conv2d(128,128,kernel_size=1,bias=False)
        self.bnatwa01_5 = nn.BatchNorm2d(128)
        self.atwa01_50 = nn.Conv2d(1,128,kernel_size=1,bias=False)
        self.atwa01_51 = nn.Conv2d(128,1,kernel_size=1,bias=False)
        self.atwp01_5 = nn.Conv2d(64,128,kernel_size=1,bias=False)
        self.bnatwp01_5 = nn.BatchNorm2d(128)
        self.atwp01_51 = nn.Conv2d(128,128,kernel_size=1,bias=False)
        self.bconv01_5 = nn.Conv2d(64,1,kernel_size=1,bias=False)
        self.bbn01_5 = nn.BatchNorm2d(1)
        self.bvconv01_5 = nn.Conv2d(64,64,kernel_size=1,bias=False)
        self.bvbn01_5 = nn.BatchNorm2d(64)
        self.bww01_5 = nn.Conv2d(64+1,1,kernel_size=1,bias=False)
        self.bwbn01_5 = nn.BatchNorm2d(1)
        self.wga01_5 = nn.Conv1d(64+2+128,128,kernel_size=1,bias=False)
        self.bnwga01_5 = nn.BatchNorm1d(128)
        self.conv01_5 = nn.Conv1d(128,128,kernel_size=1,bias=False)
        self.bn01_5c = nn.BatchNorm1d(128)
        self.sk_conv01_5 = nn.Conv1d(128+64,128,kernel_size=1,bias=False)
        self.sk_bn01_5 = nn.BatchNorm1d(128)
        self.convga01_5a = nn.Conv1d(128+128,128,kernel_size=1,bias=False)
        self.bnconvga01_5a = nn.BatchNorm1d(128)
        
        self.conv01_sc4 = nn.Conv1d(128+128,128,kernel_size=1,bias=False)
        self.bn01_sc4 = nn.BatchNorm1d(128)
        self.conv01_sc4_h = nn.Conv1d(128,128,kernel_size=1,bias=False)
        self.bn01_sc4_h = nn.BatchNorm1d(128)
        self.bn01_4 = nn.BatchNorm1d(128)
        
        self.wga01_line2 = nn.Conv1d(64+128*2,512,kernel_size=1,bias=False)
        self.bnga01_line2 = nn.BatchNorm1d(512)
        
        self.conv_d011a = nn.Conv1d(64,128,kernel_size=1,bias=False)
        self.bnconv_d011a = nn.BatchNorm1d(128)
        self.conv_d011b = nn.Conv1d(128,128,kernel_size=1,bias=False)
        self.bnconv_d011b = nn.BatchNorm1d(128)
        self.conv_d011c = nn.Conv1d(128,128,kernel_size=1,bias=False)
        self.bnconv_d011c = nn.BatchNorm1d(128)
        
        self.weight_d011 = nn.Parameter(torch.FloatTensor(128,128*3))
        init.kaiming_uniform_(self.weight_d011,a=0.2)
        self.bn_d011 = nn.BatchNorm1d(128)
        
        self.asnl01_2 = pointASNL(npoint=128,nsample=self.k[3],
                                             in_channel=128+3,b_c=64,b_c1=64,mlp_list=[64,128+1],l_mlp_list=[128,128],NL=False)
        
        self.co_conv01_6 = nn.Conv1d(3,64,kernel_size=1,bias=False)
        self.co_bn01_6 = nn.BatchNorm1d(64)
        
        self.atwq01_6m = nn.Conv2d(128,128,kernel_size=1,bias=False)
        self.atwk01_6m = nn.Conv2d(128,128,kernel_size=1,bias=False)
        
        self.atwq01_6 = nn.Conv2d(128,256,kernel_size=1,bias=False)
        self.atwkv01_6 = nn.Conv2d(128,256*2,kernel_size=1,bias=False)
        self.atwa01_6 = nn.Conv2d(256,256,kernel_size=1,bias=False)
        self.bnatwa01_6 = nn.BatchNorm2d(256)
        self.atwa01_60 = nn.Conv2d(1,256,kernel_size=1,bias=False)
        self.atwa01_61 = nn.Conv2d(256,1,kernel_size=1,bias=False)
        self.atwp01_6 = nn.Conv2d(64,256,kernel_size=1,bias=False)
        self.bnatwp01_6 = nn.BatchNorm2d(256)
        self.atwp01_61 = nn.Conv2d(256,256,kernel_size=1,bias=False)
        self.bconv01_6 = nn.Conv2d(64,1,kernel_size=1,bias=False)
        self.bbn01_6 = nn.BatchNorm2d(1)
        self.bvconv01_6 = nn.Conv2d(64,64,kernel_size=1,bias=False)
        self.bvbn01_6 = nn.BatchNorm2d(64)
        self.bww01_6 = nn.Conv2d(64+1,1,kernel_size=1,bias=False)
        self.bwbn01_6 = nn.BatchNorm2d(1)
        self.wga01_6 = nn.Conv1d(64+2+128,128,kernel_size=1,bias=False)
        self.bnwga01_6 = nn.BatchNorm1d(128)
        self.conv01_6 = nn.Conv1d(256,256,kernel_size=1,bias=False)
        self.bn01_6c = nn.BatchNorm1d(256)
        self.sk_conv01_6 = nn.Conv1d(128+64,256,kernel_size=1,bias=False)
        self.sk_bn01_6 = nn.BatchNorm1d(256)
        self.convga01_6a = nn.Conv1d(256+256,256,kernel_size=1,bias=False)
        self.bnconvga01_6a = nn.BatchNorm1d(256)
        
        self.conv01_sc5 = nn.Conv1d(256+256,256,kernel_size=1,bias=False)
        self.bn01_sc5 = nn.BatchNorm1d(256)
        self.conv01_sc5_h = nn.Conv1d(256,256,kernel_size=1,bias=False)
        self.bn01_sc5_h = nn.BatchNorm1d(256)
        self.bn01_5 = nn.BatchNorm1d(256)
        
        self.co_conv01_7 = nn.Conv1d(64,128,kernel_size=1,bias=False)
        self.co_bn01_7 = nn.BatchNorm1d(128)
        
        self.atwq01_7m = nn.Conv2d(256,256,kernel_size=1,bias=False)
        self.atwk01_7m = nn.Conv2d(256,256,kernel_size=1,bias=False)
        
        self.atwq01_7 = nn.Conv2d(256,256,kernel_size=1,bias=False)
        self.atwkv01_7 = nn.Conv2d(256,256*2,kernel_size=1,bias=False)
        self.atwa01_7 = nn.Conv2d(256,256,kernel_size=1,bias=False)
        self.bnatwa01_7 = nn.BatchNorm2d(256)
        self.atwa01_70 = nn.Conv2d(1,256,kernel_size=1,bias=False)
        self.atwa01_71 = nn.Conv2d(256,1,kernel_size=1,bias=False)
        self.atwp01_7 = nn.Conv2d(128,256,kernel_size=1,bias=False)
        self.bnatwp01_7 = nn.BatchNorm2d(256)
        self.atwp01_71 = nn.Conv2d(256,256,kernel_size=1,bias=False)
        self.bconv01_7 = nn.Conv2d(128,1,kernel_size=1,bias=False)
        self.bbn01_7 = nn.BatchNorm2d(1)
        self.bvconv01_7 = nn.Conv2d(128,128,kernel_size=1,bias=False)
        self.bvbn01_7 = nn.BatchNorm2d(128)
        self.bww01_7 = nn.Conv2d(128+1,1,kernel_size=1,bias=False)
        self.bwbn01_7 = nn.BatchNorm2d(1)
        self.wga01_7 = nn.Conv1d(128+2+256,256,kernel_size=1,bias=False)
        self.bnwga01_7 = nn.BatchNorm1d(256)
        self.conv01_7 = nn.Conv1d(256,256,kernel_size=1,bias=False)
        self.bn01_7c = nn.BatchNorm1d(256)
        self.sk_conv01_7 = nn.Conv1d(256+128,256,kernel_size=1,bias=False)
        self.sk_bn01_7 = nn.BatchNorm1d(256)
        self.convga01_7a = nn.Conv1d(256+256,256,kernel_size=1,bias=False)
        self.bnconvga01_7a = nn.BatchNorm1d(256)
        
        self.conv01_sc6 = nn.Conv1d(256+256,256,kernel_size=1,bias=False)
        self.bn01_sc6 = nn.BatchNorm1d(256)
        self.conv01_sc6_h = nn.Conv1d(256,256,kernel_size=1,bias=False)
        self.bn01_sc6_h = nn.BatchNorm1d(256)
        self.bn01_6 = nn.BatchNorm1d(256)
        
        self.wga01_line3 = nn.Conv1d(128+256*2,512,kernel_size=1,bias=False)
        self.bnga01_line3 = nn.BatchNorm1d(512)
        
        self.conv_d012a = nn.Conv1d(128,256,kernel_size=1,bias=False)
        self.bnconv_d012a = nn.BatchNorm1d(256)
        self.conv_d012b = nn.Conv1d(256,256,kernel_size=1,bias=False)
        self.bnconv_d012b = nn.BatchNorm1d(256)
        self.conv_d012c = nn.Conv1d(256,256,kernel_size=1,bias=False)
        self.bnconv_d012c = nn.BatchNorm1d(256)
        
        self.weight_d012 = nn.Parameter(torch.FloatTensor(256,256*3))
        init.kaiming_uniform_(self.weight_d012,a=0.2)
        self.bn_d012 = nn.BatchNorm1d(256)
        
        self.fp2 = PointNetFeaturePropagation(in_channel=128+512*2+256+512*2, mlp=[256,128])
        
        self.fp1 = PointNetFeaturePropagation(in_channel=64+512*2+128, mlp=[128, 128])
        
        self.conv1 = nn.Conv1d(128+3+3+16, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(0.4)
        self.conv2 = nn.Conv1d(128, num_part, 1)
    
    def forward(self,points,cls_label,as_neighbor=11):
        batch_size = points.shape[0]
        l0_xyz = points
        l0_points = points
    
        #graph_conv1
        in_gc = points.view(batch_size,-1,self.num_points,1)
        
        f_abs,diff = get_graph_feature(x=points,k=as_neighbor)
        basevactor = F.leaky_relu(self.bvbn01_1(self.bvconv01_1(diff.permute(0,3,1,2))),0.2)#[B,C,N,k]
        basevactor = torch.mean(basevactor,dim=3).view(batch_size,-1,self.num_points,1)#[B,C,N,1]
        basevactor = basevactor.permute(0,2,3,1)#[B,N,1,C]
        distent1 = torch.sum(basevactor ** 2,dim=3) 
        distent2 = torch.sum(diff ** 2,dim=3) 
        inn_product = basevactor.matmul(diff.permute(0,1,3,2)).view(batch_size,self.num_points,-1)
        cos = (inn_product / torch.sqrt((distent1 * distent2) + 1e-10)).view(batch_size,self.num_points,as_neighbor-1,1)
        #cos = torch.mean(cos,dim=2).transpose(2,1)#[B,C,N]
        diff = F.leaky_relu(self.bbn01_1(self.bconv01_1(diff.permute(0,3,1,2))),0.2).permute(0,2,3,1)#[B,N,k,1]
        #diff = torch.mean(diff,dim=3)#[B,C,N]
        agg_info1 = torch.cat((f_abs,diff),dim=3)#[B,N,k,C]
        t = agg_info1.permute(0,3,1,2)#[B,C,N,k]
        t = F.leaky_relu(self.bwbn01_1(self.bww01_1(t)),0.2).permute(0,2,3,1)#[B,N,k,1]
        agg_info1 = torch.cat((f_abs,diff,t),dim=3)#[B,N,k,C]
        agg_info1 = (cos * agg_info1).permute(0,3,1,2)#[B,C,N,k]
        agg_info1 = torch.sum(agg_info1,dim=-1)#[B,C,N]
        agg_info1 = torch.cat([points,agg_info1],dim=1)#[B,C,N]
        agg_info1 = F.relu(self.bnwga01(self.wga01(agg_info1)))#[B,C,N]
        
        nS,p,diff = get_graph_feature_A(x=agg_info1,xyz=points,k=self.k[0])
        new_points = torch.cat((diff.permute(0,3,1,2),in_gc.repeat(1,1,1,self.k[0])),dim=1)
        new_points = torch.max(new_points,dim=-1)[0]
        new_points = F.relu(self.sk_bn01_1(self.sk_conv01_1(new_points)))
        in_gc_t = agg_info1.view(batch_size,-1,self.num_points,1)#[B,C,N,1]
        node = nS.permute(0,3,1,2)#[B,C,N,k]
        Qm = self.atwq01_1m(in_gc_t)#[B,C,N,1]
        Km = self.atwk01_1m(node)#[B,C,N,k]
        Am = torch.matmul(Qm.permute(0,2,3,1),Km.permute(0,2,1,3))#[B,N,1,k]
        at = self.atwa01_10(Am.permute(0,2,1,3))#[B,C,N,k]
        Q = self.atwq01_1(in_gc_t)#[B,C,N,1]
        KV = self.atwkv01_1(node)#[B,C,N,k]
        K = KV[:,:64,:,:]
        V = KV[:,64:,:,:]
        QK = Q - K#[B,C,N,k]
        p = p.permute(0,3,1,2)#[B,C,N,k]
        p = F.relu(self.bnatwp01_1(self.atwp01_1(p)))#[B,C,N,k]
        p = self.atwp01_11(p)#[B,C,N,k]
        QK = QK + p + at#[B,C,N,k]
        A = F.relu(self.bnatwa01_1(self.atwa01_1(QK)))#[B,C,N,k]
        A = self.atwa01_11(A)#[B,1,N,k]
        A = A/(64**0.5)
        A = F.softmax(A,dim=-1)
        node1 = A * (V+p)
        node1 = torch.sum(node1,dim=-1)#[B,C,N]
        node1 = F.leaky_relu(self.bn01_1c(self.conv01_1(node1)),0.2)
        node1 = torch.cat((new_points,node1),dim=1)
        self_f0 = F.leaky_relu(self.bnconvga01_1a(self.convga01_1a(node1)),0.2)
        new_points0_sc_t = self.bn01_sc0(self.conv01_sc0(node1))
        new_points0_t = self.bn01_sc0_h(self.conv01_sc0_h(self_f0))
        new_points0_t = F.leaky_relu(self.bn01_0(new_points0_sc_t+new_points0_t),0.2)
        
        node_f01_2 = new_points0_t
        co = F.relu(self.co_bn01_2(self.co_conv01_2(points)))
        in_gc = node_f01_2.view(batch_size,-1,self.num_points,1)
        
        f_abs,diff = get_graph_feature(x=co,k=as_neighbor)
        basevactor = F.leaky_relu(self.bvbn01_2(self.bvconv01_2(diff.permute(0,3,1,2))),0.2)#[B,C,N,k]
        basevactor = torch.mean(basevactor,dim=3).view(batch_size,-1,self.num_points,1)#[B,C,N,1]
        basevactor = basevactor.permute(0,2,3,1)#[B,N,1,C]
        distent1 = torch.sum(basevactor ** 2,dim=3) 
        distent2 = torch.sum(diff ** 2,dim=3) 
        inn_product = basevactor.matmul(diff.permute(0,1,3,2)).view(batch_size,self.num_points,-1)
        cos = (inn_product / torch.sqrt((distent1 * distent2) + 1e-10)).view(batch_size,self.num_points,as_neighbor-1,1)
        #cos = torch.mean(cos,dim=2).transpose(2,1)#[B,C,N]
        diff = F.leaky_relu(self.bbn01_2(self.bconv01_2(diff.permute(0,3,1,2))),0.2).permute(0,2,3,1)#[B,N,k,1]
        #diff = torch.mean(diff,dim=3)#[B,C,N]
        agg_info1 = torch.cat((f_abs,diff),dim=3)#[B,N,k,C]
        t = agg_info1.permute(0,3,1,2)#[B,C,N,k]
        t = F.leaky_relu(self.bwbn01_2(self.bww01_2(t)),0.2).permute(0,2,3,1)#[B,N,k,1]
        agg_info1 = torch.cat((f_abs,diff,t),dim=3)#[B,N,k,C]
        agg_info1 = (cos * agg_info1).permute(0,3,1,2)#[B,C,N,k]
        agg_info1 = torch.sum(agg_info1,dim=-1)#[B,C,N]
        agg_info1 = torch.cat([node_f01_2,agg_info1],dim=1)#[B,C,N]
        agg_info1 = F.relu(self.bnwga01_2(self.conv01_self(agg_info1)))#[B,C,N]
        
        nS,p,diff = get_graph_feature_A(x=agg_info1,xyz=co,k=self.k[0])
        new_points = torch.cat((diff.permute(0,3,1,2),in_gc.repeat(1,1,1,self.k[0])),dim=1)
        new_points = torch.max(new_points,dim=-1)[0]
        new_points = F.relu(self.sk_bn01_2(self.sk_conv01_2(new_points)))
        in_gc_t = agg_info1.view(batch_size,-1,self.num_points,1)#[B,C,N,1]
        node = nS.permute(0,3,1,2)#[B,C,N,k]
        Qm = self.atwq01_2m(in_gc_t)#[B,C,N,1]
        Km = self.atwk01_2m(node)#[B,C,N,k]
        Am = torch.matmul(Qm.permute(0,2,3,1),Km.permute(0,2,1,3))#[B,N,1,k]
        at = self.atwa01_20(Am.permute(0,2,1,3))#[B,C,N,k]
        Q = self.atwq01_2(in_gc_t)#[B,C,N,1]
        KV = self.atwkv01_2(node)#[B,C,N,k]
        K = KV[:,:64,:,:]
        V = KV[:,64:,:,:]
        QK = Q - K
        p = p.permute(0,3,1,2)#[B,C,N,k]
        p = F.relu(self.bnatwp01_2(self.atwp01_2(p)))#[B,C,N,k]
        p = self.atwp01_21(p)#[B,C,N,k]
        QK = QK + p + at
        A = F.relu(self.bnatwa01_2(self.atwa01_2(QK)))#[B,C,N,k]
        A = self.atwa01_21(A)#[B,1,N,k]
        A = A/(64**0.5)
        A = F.softmax(A,dim=-1)
        node1 = A * (V+p)
        node1 = torch.sum(node1,dim=-1)#[B,C,N]
        node1 = F.leaky_relu(self.bn01_2c(self.conv01_2(node1)),0.2)
        node1 = torch.cat((new_points,node1),dim=1)
        self_f0 = F.leaky_relu(self.bnconvga01_2a(self.convga01_2a(node1)),0.2)
        new_points0_1t = self.bn01_sc1(self.conv01_sc1(node1))
        new_points0 = self.bn01_sc1_h(self.conv01_sc1_h(self_f0))
        new_points0_1 = F.leaky_relu(self.bn01_1(new_points0_1t+new_points0),0.2)
        
        new_points10 = torch.cat((node_f01_2,new_points0_1),dim=1)
        new_points10 = self.bnga01_line1(self.wga01_line1(new_points10))
        x_max00 = F.adaptive_max_pool1d(new_points10,1).view(batch_size,-1)
        x_avg00 = F.adaptive_avg_pool1d(new_points10,1).view(batch_size,-1)
        x_new10 = torch.cat((x_max00,x_avg00),dim=1)
        
        points0 = self.bnconv_d01a(self.conv_d01a(points))
        new_points0_t = self.bnconv_d01b(self.conv_d01b(node_f01_2))
        new_points0_1 = self.bnconv_d01c(self.conv_d01c(new_points0_1))
        new_points10 = torch.cat((points0,new_points0_t,new_points0_1),dim=1)
        new_points0_pool = F.relu(self.bn_d01(self.weight_d01.matmul(new_points10)))
        l0_points = new_points0_pool
        
        l1_xyz,l1_points,pool_idx = self.asnl01_1(l0_xyz,l0_points)

        new_points0_2 = l1_points
        
        node_f01_4 = new_points0_2
        co = F.relu(self.co_bn01_4(self.co_conv01_4(l1_xyz)))
        in_gc = node_f01_4.view(batch_size,-1,512,1)
        
        f_abs,diff = get_graph_feature(x=co,k=as_neighbor)
        basevactor = F.leaky_relu(self.bvbn01_4(self.bvconv01_4(diff.permute(0,3,1,2))),0.2)#[B,C,N,k]
        basevactor = torch.mean(basevactor,dim=3).view(batch_size,-1,512,1)#[B,C,N,1]
        basevactor = basevactor.permute(0,2,3,1)#[B,N,1,C]
        distent1 = torch.sum(basevactor ** 2,dim=3) 
        distent2 = torch.sum(diff ** 2,dim=3) 
        inn_product = basevactor.matmul(diff.permute(0,1,3,2)).view(batch_size,512,-1)
        cos = (inn_product / torch.sqrt((distent1 * distent2) + 1e-10)).view(batch_size,512,as_neighbor-1,1)
        #cos = torch.mean(cos,dim=2).transpose(2,1)#[B,C,N]
        diff = F.leaky_relu(self.bbn01_4(self.bconv01_4(diff.permute(0,3,1,2))),0.2).permute(0,2,3,1)#[B,N,k,1]
        #diff = torch.mean(diff,dim=3)#[B,C,N]
        agg_info1 = torch.cat((f_abs,diff),dim=3)#[B,N,k,C]
        t = agg_info1.permute(0,3,1,2)#[B,C,N,k]
        t = F.leaky_relu(self.bwbn01_4(self.bww01_4(t)),0.2).permute(0,2,3,1)#[B,N,k,1]
        agg_info1 = torch.cat((f_abs,diff,t),dim=3)#[B,N,k,C]
        agg_info1 = (cos * agg_info1).permute(0,3,1,2)#[B,C,N,k]
        agg_info1 = torch.sum(agg_info1,dim=-1)#[B,C,N]
        agg_info1 = torch.cat([node_f01_4,agg_info1],dim=1)#[B,C,N]
        agg_info1 = F.relu(self.bnwga01_4(self.wga01_4(agg_info1)))#[B,C,N]
        
        nS,p,diff = get_graph_feature_A(x=agg_info1,xyz=co,k=self.k[0])
        new_points = torch.cat((diff.permute(0,3,1,2),in_gc.repeat(1,1,1,self.k[0])),dim=1)
        new_points = torch.max(new_points,dim=-1)[0]
        new_points = F.relu(self.sk_bn01_4(self.sk_conv01_4(new_points)))
        in_gc_t = agg_info1.view(batch_size,-1,512,1)#[B,C,N,1]
        node = nS.permute(0,3,1,2)#[B,C,N,k]
        Qm = self.atwq01_4m(in_gc_t)#[B,C,N,1]
        Km = self.atwk01_4m(node)#[B,C,N,k]
        Am = torch.matmul(Qm.permute(0,2,3,1),Km.permute(0,2,1,3))#[B,N,1,k]
        at = self.atwa01_40(Am.permute(0,2,1,3))#[B,C,N,k]
        Q = self.atwq01_4(in_gc_t)#[B,C,N,1]
        KV = self.atwkv01_4(node)#[B,C,N,k]
        K = KV[:,:128,:,:]
        V = KV[:,128:,:,:]
        QK = Q - K
        p = p.permute(0,3,1,2)#[B,C,N,k]
        p = F.relu(self.bnatwp01_4(self.atwp01_4(p)))#[B,C,N,k]
        p = self.atwp01_41(p)#[B,C,N,k]
        QK = QK + p + at 
        A = F.relu(self.bnatwa01_4(self.atwa01_4(QK)))#[B,C,N,k]
        A = self.atwa01_41(A)#[B,1,N,k]
        A = A/(128**0.5)
        A = F.softmax(A,dim=-1)
        node1 = A * (V+p)
        node1 = torch.sum(node1,dim=-1)#[B,C,N]
        node1 = F.leaky_relu(self.bn01_4c(self.conv01_4(node1)),0.2)
        node1 = torch.cat((new_points,node1),dim=1)
        self_f0 = F.leaky_relu(self.bnconvga01_4a(self.convga01_4a(node1)),0.2)
        new_points0 = self.bn01_sc3(self.conv01_sc3(node1))
        new_points0_sc = self.bn01_sc3_h(self.conv01_sc3_h(self_f0))
        new_points0 = F.leaky_relu(self.bn01_3(new_points0+new_points0_sc),0.2)
        
        node_f01_5 = new_points0
        co = F.relu(self.co_bn01_5(self.co_conv01_5(co)))
        in_gc = node_f01_5.view(batch_size,-1,512,1)
        
        f_abs,diff = get_graph_feature(x=co,k=as_neighbor)
        basevactor = F.leaky_relu(self.bvbn01_5(self.bvconv01_5(diff.permute(0,3,1,2))),0.2)#[B,C,N,k]
        basevactor = torch.mean(basevactor,dim=3).view(batch_size,-1,512,1)#[B,C,N,1]
        basevactor = basevactor.permute(0,2,3,1)#[B,N,1,C]
        distent1 = torch.sum(basevactor ** 2,dim=3) 
        distent2 = torch.sum(diff ** 2,dim=3) 
        inn_product = basevactor.matmul(diff.permute(0,1,3,2)).view(batch_size,512,-1)
        cos = (inn_product / torch.sqrt((distent1 * distent2) + 1e-10)).view(batch_size,512,as_neighbor-1,1)
        #cos = torch.mean(cos,dim=2).transpose(2,1)#[B,C,N]
        diff = F.leaky_relu(self.bbn01_5(self.bconv01_5(diff.permute(0,3,1,2))),0.2).permute(0,2,3,1)#[B,N,k,1]
        #diff = torch.mean(diff,dim=3)#[B,C,N]
        agg_info1 = torch.cat((f_abs,diff),dim=3)#[B,N,k,C]
        t = agg_info1.permute(0,3,1,2)#[B,C,N,k]
        t = F.leaky_relu(self.bwbn01_5(self.bww01_5(t)),0.2).permute(0,2,3,1)#[B,N,k,1]
        agg_info1 = torch.cat((f_abs,diff,t),dim=3)#[B,N,k,C]
        agg_info1 = (cos * agg_info1).permute(0,3,1,2)#[B,C,N,k]
        agg_info1 = torch.sum(agg_info1,dim=-1)#[B,C,N]
        agg_info1 = torch.cat([node_f01_5,agg_info1],dim=1)#[B,C,N]
        agg_info1 = F.relu(self.bnwga01_5(self.wga01_5(agg_info1)))#[B,C,N]
        
        nS,p,diff = get_graph_feature_A(x=agg_info1,xyz=co,k=self.k[0])
        new_points = torch.cat((diff.permute(0,3,1,2),in_gc.repeat(1,1,1,self.k[0])),dim=1)
        new_points = torch.max(new_points,dim=-1)[0]
        new_points = F.relu(self.sk_bn01_5(self.sk_conv01_5(new_points)))
        in_gc_t = agg_info1.view(batch_size,-1,512,1)#[B,C,N,1]
        node = nS.permute(0,3,1,2)#[B,C,N,k]
        Qm = self.atwq01_5m(in_gc_t)#[B,C,N,1]
        Km = self.atwk01_5m(node)#[B,C,N,k]
        Am = torch.matmul(Qm.permute(0,2,3,1),Km.permute(0,2,1,3))#[B,N,1,k]
        at = self.atwa01_50(Am.permute(0,2,1,3))#[B,C,N,k]
        Q = self.atwq01_5(in_gc_t)#[B,C,N,1]
        KV = self.atwkv01_5(node)#[B,C,N,k]
        K = KV[:,:128,:,:]
        V = KV[:,128:,:,:]
        QK = Q - K
        p = p.permute(0,3,1,2)#[B,C,N,k]
        p = F.relu(self.bnatwp01_5(self.atwp01_5(p)))#[B,C,N,k]
        p = self.atwp01_51(p)#[B,C,N,k]
        QK = QK + p + at
        A = F.relu(self.bnatwa01_5(self.atwa01_5(QK)))#[B,C,N,k]
        A = self.atwa01_51(A)#[B,1,N,k]
        A = A/(128**0.5)
        A = F.softmax(A,dim=-1)
        node1 = A * (V+p)
        node1 = torch.sum(node1,dim=-1)#[B,C,N]
        node1 = F.leaky_relu(self.bn01_5c(self.conv01_5(node1)),0.2)
        node1 = torch.cat((new_points,node1),dim=1)
        self_f0 = F.leaky_relu(self.bnconvga01_5a(self.convga01_5a(node1)),0.2)
        new_points0_3t = self.bn01_sc4(self.conv01_sc4(node1))
        new_points0_sc = self.bn01_sc4_h(self.conv01_sc4_h(self_f0))
        new_points0_3 = F.leaky_relu(self.bn01_4(new_points0_3t+new_points0_sc),0.2)
        
        new_points1 = torch.cat((node_f01_4,node_f01_5,new_points0_3),dim=1)
        new_points1 = self.bnga01_line2(self.wga01_line2(new_points1))
        x_max01 = F.adaptive_max_pool1d(new_points1,1).view(batch_size,-1)
        x_avg01 = F.adaptive_avg_pool1d(new_points1,1).view(batch_size,-1)
        x_new11 = torch.cat((x_max01,x_avg01),dim=1)
        
        new_points0_2 = self.bnconv_d011a(self.conv_d011a(node_f01_4))
        new_points0 = self.bnconv_d011b(self.conv_d011b(node_f01_5))
        new_points0_3 = self.bnconv_d011c(self.conv_d011c(new_points0_3))
        new_points1 = torch.cat((new_points0_2,new_points0,new_points0_3),dim=1)
        new_points0_pool1 = F.relu(self.bn_d011(self.weight_d011.matmul(new_points1)))
        l1_points = new_points0_pool1
        
        l2_xyz,l2_points,pool_idx = self.asnl01_2(l1_xyz,l1_points)
        
        new_points0_4 = l2_points
        
        node_f01_6 = new_points0_4
        co = F.relu(self.co_bn01_6(self.co_conv01_6(l2_xyz)))
        in_gc = node_f01_6.view(batch_size,-1,128,1)
        
        f_abs,diff = get_graph_feature(x=co,k=as_neighbor)
        basevactor = F.leaky_relu(self.bvbn01_6(self.bvconv01_6(diff.permute(0,3,1,2))),0.2)#[B,C,N,k]
        basevactor = torch.mean(basevactor,dim=3).view(batch_size,-1,128,1)#[B,C,N,1]
        basevactor = basevactor.permute(0,2,3,1)#[B,N,1,C]
        distent1 = torch.sum(basevactor ** 2,dim=3) 
        distent2 = torch.sum(diff ** 2,dim=3) 
        inn_product = basevactor.matmul(diff.permute(0,1,3,2)).view(batch_size,128,-1)
        cos = (inn_product / torch.sqrt((distent1 * distent2) + 1e-10)).view(batch_size,128,as_neighbor-1,1)
        #cos = torch.mean(cos,dim=2).transpose(2,1)#[B,C,N]
        diff = F.leaky_relu(self.bbn01_6(self.bconv01_6(diff.permute(0,3,1,2))),0.2).permute(0,2,3,1)#[B,N,k,1]
        #diff = torch.mean(diff,dim=3)#[B,C,N]
        agg_info1 = torch.cat((f_abs,diff),dim=3)#[B,N,k,C]
        t = agg_info1.permute(0,3,1,2)#[B,C,N,k]
        t = F.leaky_relu(self.bwbn01_6(self.bww01_6(t)),0.2).permute(0,2,3,1)#[B,N,k,1]
        agg_info1 = torch.cat((f_abs,diff,t),dim=3)#[B,N,k,C]
        agg_info1 = (cos * agg_info1).permute(0,3,1,2)#[B,C,N,k]
        agg_info1 = torch.sum(agg_info1,dim=-1)#[B,C,N]
        agg_info1 = torch.cat([node_f01_6,agg_info1],dim=1)#[B,C,N]
        agg_info1 = F.relu(self.bnwga01_6(self.wga01_6(agg_info1)))#[B,C,N]
        
        nS,p,diff = get_graph_feature_A(x=agg_info1,xyz=co,k=self.k[0])
        new_points = torch.cat((diff.permute(0,3,1,2),in_gc.repeat(1,1,1,self.k[0])),dim=1)
        new_points = torch.max(new_points,dim=-1)[0]
        new_points = F.relu(self.sk_bn01_6(self.sk_conv01_6(new_points)))
        in_gc_t = agg_info1.view(batch_size,-1,128,1)#[B,C,N,1]
        node = nS.permute(0,3,1,2)#[B,C,N,k]
        Qm = self.atwq01_6m(in_gc_t)#[B,C,N,1]
        Km = self.atwk01_6m(node)#[B,C,N,k]
        Am = torch.matmul(Qm.permute(0,2,3,1),Km.permute(0,2,1,3))#[B,N,1,k]
        at = self.atwa01_60(Am.permute(0,2,1,3))#[B,C,N,k]
        Q = self.atwq01_6(in_gc_t)#[B,C,N,1]
        KV = self.atwkv01_6(node)#[B,C,N,k]
        K = KV[:,:256,:,:]
        V = KV[:,256:,:,:]
        QK = Q - K
        p = p.permute(0,3,1,2)#[B,C,N,k]
        p = F.relu(self.bnatwp01_6(self.atwp01_6(p)))#[B,C,N,k]
        p = self.atwp01_61(p)#[B,C,N,k]
        QK = QK + p + at 
        A = F.relu(self.bnatwa01_6(self.atwa01_6(QK)))#[B,C,N,k]
        A = self.atwa01_61(A)#[B,1,N,k]
        A = A/(256**0.5)
        A = F.softmax(A,dim=-1)
        node1 = A * (V+p)
        node1 = torch.sum(node1,dim=-1)#[B,C,N]
        node1 = F.leaky_relu(self.bn01_6c(self.conv01_6(node1)),0.2)
        node1 = torch.cat((new_points,node1),dim=1)
        self_f0 = F.leaky_relu(self.bnconvga01_6a(self.convga01_6a(node1)),0.2)
        new_points0_5 = self.bn01_sc5(self.conv01_sc5(node1))
        new_points0_sc = self.bn01_sc5_h(self.conv01_sc5_h(self_f0))
        new_points0_5 = F.leaky_relu(self.bn01_5(new_points0_5+new_points0_sc),0.2)
        
        node_f01_7 = new_points0_5
        co = F.relu(self.co_bn01_7(self.co_conv01_7(co)))
        in_gc = node_f01_7.view(batch_size,-1,128,1)
        
        f_abs,diff = get_graph_feature(x=co,k=as_neighbor)
        basevactor = F.leaky_relu(self.bvbn01_7(self.bvconv01_7(diff.permute(0,3,1,2))),0.2)#[B,C,N,k]
        basevactor = torch.mean(basevactor,dim=3).view(batch_size,-1,128,1)#[B,C,N,1]
        basevactor = basevactor.permute(0,2,3,1)#[B,N,1,C]
        distent1 = torch.sum(basevactor ** 2,dim=3) 
        distent2 = torch.sum(diff ** 2,dim=3) 
        inn_product = basevactor.matmul(diff.permute(0,1,3,2)).view(batch_size,128,-1)
        cos = (inn_product / torch.sqrt((distent1 * distent2) + 1e-10)).view(batch_size,128,as_neighbor-1,1)
        #cos = torch.mean(cos,dim=2).transpose(2,1)#[B,C,N]
        diff = F.leaky_relu(self.bbn01_7(self.bconv01_7(diff.permute(0,3,1,2))),0.2).permute(0,2,3,1)#[B,N,k,1]
        #diff = torch.mean(diff,dim=3)#[B,C,N]
        agg_info1 = torch.cat((f_abs,diff),dim=3)#[B,N,k,C]
        t = agg_info1.permute(0,3,1,2)#[B,C,N,k]
        t = F.leaky_relu(self.bwbn01_7(self.bww01_7(t)),0.2).permute(0,2,3,1)#[B,N,k,1]
        agg_info1 = torch.cat((f_abs,diff,t),dim=3)#[B,N,k,C]
        agg_info1 = (cos * agg_info1).permute(0,3,1,2)#[B,C,N,k]
        agg_info1 = torch.sum(agg_info1,dim=-1)#[B,C,N]
        agg_info1 = torch.cat([node_f01_7,agg_info1],dim=1)#[B,C,N]
        agg_info1 = F.relu(self.bnwga01_7(self.wga01_7(agg_info1)))#[B,C,N]
        
        nS,p,diff = get_graph_feature_A(x=agg_info1,xyz=co,k=self.k[0])
        new_points = torch.cat((diff.permute(0,3,1,2),in_gc.repeat(1,1,1,self.k[0])),dim=1)
        new_points = torch.max(new_points,dim=-1)[0]
        new_points = F.relu(self.sk_bn01_7(self.sk_conv01_7(new_points)))
        in_gc_t = agg_info1.view(batch_size,-1,128,1)#[B,C,N,1]
        node = nS.permute(0,3,1,2)#[B,C,N,k]
        Qm = self.atwq01_7m(in_gc_t)#[B,C,N,1]
        Km = self.atwk01_7m(node)#[B,C,N,k]
        Am = torch.matmul(Qm.permute(0,2,3,1),Km.permute(0,2,1,3))#[B,N,1,k]
        at = self.atwa01_70(Am.permute(0,2,1,3))#[B,C,N,k]
        Q = self.atwq01_7(in_gc_t)#[B,C,N,1]
        KV = self.atwkv01_7(node)#[B,C,N,k]
        K = KV[:,:256,:,:]
        V = KV[:,256:,:,:]
        QK = Q - K
        p = p.permute(0,3,1,2)#[B,C,N,k]
        p = F.relu(self.bnatwp01_7(self.atwp01_7(p)))#[B,C,N,k]
        p = self.atwp01_71(p)#[B,C,N,k]
        QK = QK + p + at
        A = F.relu(self.bnatwa01_7(self.atwa01_7(QK)))#[B,C,N,k]
        A = self.atwa01_71(A)#[B,1,N,k]
        A = A/(256**0.5)
        A = F.softmax(A,dim=-1)
        node1 = A * (V+p)
        node1 = torch.sum(node1,dim=-1)#[B,C,N]
        node1 = F.leaky_relu(self.bn01_7c(self.conv01_7(node1)),0.2)
        node1 = torch.cat((new_points,node1),dim=1)
        self_f0 = F.leaky_relu(self.bnconvga01_7a(self.convga01_7a(node1)),0.2)
        new_points0_6t = self.bn01_sc6(self.conv01_sc6(node1))
        new_points0_sc = self.bn01_sc6_h(self.conv01_sc6_h(self_f0))
        new_points0_6 = F.leaky_relu(self.bn01_6(new_points0_6t+new_points0_sc),0.2)
        
        new_points11 = torch.cat((node_f01_6,node_f01_7,new_points0_6),dim=1)
        new_points11 = self.bnga01_line3(self.wga01_line3(new_points11))
        x_max11 = F.adaptive_max_pool1d(new_points11,1).view(batch_size,-1)
        x_avg11 = F.adaptive_avg_pool1d(new_points11,1).view(batch_size,-1)
        x_new12 = torch.cat((x_max11,x_avg11),dim=1)
        
        new_points0_4 = self.bnconv_d012a(self.conv_d012a(node_f01_6))
        new_points0_5 = self.bnconv_d012b(self.conv_d012b(node_f01_7))
        new_points0_6 = self.bnconv_d012c(self.conv_d012c(new_points0_6))
        new_points11 = torch.cat((new_points0_4,new_points0_5,new_points0_6),dim=1)
        new_points0_pool2 = F.relu(self.bn_d012(self.weight_d012.matmul(new_points11)))
        l2_points = new_points0_pool2
        
        # Feature Propagation layers
        l2_points = torch.cat((l2_points,x_new12.view(batch_size,-1,1).repeat(1,1,128)),dim=1)
        
        l1_points = torch.cat((l1_points,x_new11.view(batch_size,-1,1).repeat(1,1,512)),dim=1)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        
        l0_points = torch.cat((l0_points,x_new10.view(batch_size,-1,1).repeat(1,1,self.num_points)),dim=1)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)
        
        # FC layers
        cls_label_one_hot = cls_label.view(batch_size,16,1).repeat(1,1,self.num_points)
        l0_points = torch.cat((torch.cat([cls_label_one_hot,l0_xyz,points],1),l0_points),dim=1)
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop(feat)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        
        return x

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)

        return total_loss
import torch,math
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter, scatter_sum

# Deeply Explore Mist by Evolution model  得密
class DEME(nn.Module):
    def __init__(self,esm_pretrain_model,saprot_pretrain_model):
        super(DEME,self).__init__()
        self.esm_pretrain_model = esm_pretrain_model
        self.saprot_pretrain_model = saprot_pretrain_model
        self.w = nn.Linear(2,1,bias=False)

    def forward(self,data):
        esm_x,padding_mask = to_dense_batch(data['protein'].x,data['protein'].batch)
        dense_batch,_ = to_dense_batch(data['protein'].batch,data['protein'].batch)
        esm_x[~padding_mask] = 1
        
        esm_logits = self.esm_pretrain_model(esm_x)['logits']
        esm_mask = esm_x==32

        esm_mask_logits = esm_logits[esm_mask]
        esm_mask_batch = dense_batch[esm_mask]

        esm_logits1 = esm_mask_logits.gather(dim=1,index=data.esm_token_idx[:,[0]])
        esm_logits2 = esm_mask_logits.gather(dim=1,index=data.esm_token_idx[:,[1]])

    
        esm_logits_sum1 = scatter_sum(esm_logits1, esm_mask_batch, dim=0)
        esm_logits_sum2 = scatter_sum(esm_logits2, esm_mask_batch, dim=0)

        return esm_logits_sum2-esm_logits_sum1
        
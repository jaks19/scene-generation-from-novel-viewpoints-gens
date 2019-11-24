from __future__ import print_function
import json
import numpy as np
from tqdm import tqdm as Tqdm

import torch
from torch.nn import ParameterList
import torch.nn as nn
from GQN.modules import Pool

import Utils.global_vars as glo

class GQN_Composer(nn.Module):
  def __init__(self, num_copies=0):
    super(GQN_Composer, self).__init__()
    self.embedder = Pool()
    self.num_copies = num_copies
    
  def forward(self, view_frames, view_poses, query_poses):
    bs, num_views_per_query = view_frames.shape[:2]
    r = view_frames.new_zeros((bs, 254))
    # import pdb; pdb.set_trace()
    for k in range(num_views_per_query): 
      r += self.embedder(view_frames[:, k], view_poses[:, k]).squeeze(3).squeeze(2)
    
    # Add pose coordinates to head of embeddings 
    # First two added to replicate behaviour of GEN and include x,y even when num_copies=0
    embeddings = torch.cat([query_poses[:,:,:2]] + [query_poses]*self.num_copies + [r.unsqueeze(1).repeat(1, query_poses.shape[1], 1)], dim=2)

    assert(embeddings.shape == (bs, query_poses.shape[1], 256 + (self.num_copies * 7)))
    return embeddings

  def refresh_structure(self, scenes_per_dim_train, shift_train):
    # No structure in GQN composer, exists to meet specs for a composer
    return
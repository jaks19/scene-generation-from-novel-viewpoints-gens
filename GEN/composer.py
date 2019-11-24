'''
'Graph Element Networks' Composition
'''

# Use passed in parameters' devices as reference as the entry point is the forward method
# and this would make sure models are also on same devices
from __future__ import print_function
import json
import numpy as np
from tqdm import tqdm as Tqdm

import torch
from torch.nn import ParameterList
import torch.nn as nn

from GEN.structure import GraphStructure

from GQN.modules import Pool
from GEN.modules import NodeModule, EdgeModule

import Utils.global_vars as glo

class GEN_Composer(nn.Module):
  def __init__(self, num_copies=0):
    super(GEN_Composer, self).__init__()
    
    self.structure = GraphStructure()
    self.node_module = NodeModule()
    self.edge_module = EdgeModule()
    self.embedder = Pool()

    # Parameters for when we concat the pose coordinates to the extracted embedding
    self.num_copies = num_copies


  # Refreshed self.structure when graph shape changes
  def refresh_structure(self, num_scenes_per_dim, shift):
    self.structure = GraphStructure(num_scenes_per_dim=num_scenes_per_dim, shift=shift)
    return
  
  # Preprocesses raw frames and poses into embeddings which then fit into the node hidden states
  # using an attention score interpolated from the structure of the graph and position of the inputs
  def inp_to_graph_inp(self, view_frames, view_poses):
    batch_size, num_views_per_batch = view_frames.shape[:2]

    # Squash batch dim and num of frames to have only one dimension with num of frames
    frames_reshaped = view_frames.reshape(batch_size * num_views_per_batch, 3, glo.IMG_SIZE, glo.IMG_SIZE)
    poses_reshaped = view_poses.reshape(batch_size * num_views_per_batch, 7)
    assert(frames_reshaped.device == poses_reshaped.device)

    embeddings = self.embedder.forward(frames_reshaped, poses_reshaped)
    assert(embeddings.shape == (batch_size * num_views_per_batch, 254, 1, 1))
    
    # Regain batch dimension and flatten
    embeddings = embeddings.reshape(batch_size, num_views_per_batch, 254)

    # Populate state of each node (for each batch)
    # 1) Organize node_positions
    node_positions = self.structure.node_positions.unsqueeze(0).repeat((batch_size,1,1)).to(view_frames.device)
    assert(node_positions.shape == (batch_size, self.structure.num_nodes, 2))

    # 2) Get weighted embeddings per node
    scores, more = self.structure.get_interpolation_coordinates(view_poses)
    assert(scores.shape == (batch_size, num_views_per_batch, self.structure.num_nodes))
    # Reshape for bmm
    scores = scores.permute(0,2,1)

    # Get weighted embeddings
    weighted = torch.bmm(scores, embeddings)
    assert(weighted.shape == (batch_size, self.structure.num_nodes, 254))

    # 3) Concat node positions to the weighted embeddings to get tail shape 2+254
    inp = torch.cat([node_positions, weighted], dim=2)
    assert(inp.shape == (batch_size, self.structure.num_nodes, 256))

    extra = True
    if extra: return inp, more

    return inp


  # Given node hidden states, extrapolates query embeddings from the graph
  # based on the positions of the queries
  def graph_out_to_out(self, nodes_states, query_poses):
    # Apply each corresponding integrator to each node
    # Sum the results of the integrations
    # Apply final module
    batch_size = query_poses.shape[0]
    poses_per_scene = query_poses.shape[1]
    pose_dimension = query_poses.shape[2]

    # F is the nodes hidden states per scene (per room or per maze configuration)
    assert(nodes_states.shape == (batch_size, self.structure.num_nodes, self.structure.node_dim))

    # Interpolation tensor
    # Unsqueeze index 1 as query poses lack that dimension (always 1 query used)
    attn, more = self.structure.get_interpolation_coordinates(query_poses)
    assert(attn.shape == (batch_size, poses_per_scene, self.structure.num_nodes))

    # Extract the weighted sum of node hidden states to decode, for each pose 
    extraction = torch.bmm(attn, nodes_states)
    assert(extraction.shape == (batch_size, poses_per_scene, self.structure.node_dim))

    # Assert that the (x.y) coordinates were properly extracted for each node hidden state
    if not (torch.max(torch.abs(extraction[:,:,:2] - query_poses[:,:,:2])) < 0.05):
      print('Extraction-Poses assertion failed', torch.mean(torch.abs(extraction[:,:,:2] - query_poses[:,:,:2])))

    extraction = torch.cat([query_poses]*self.num_copies + [extraction], dim=2)
    assert(extraction.shape == (batch_size, poses_per_scene, self.structure.node_dim + (self.num_copies * 7)))
    
    return extraction, more


  # Given raw frames and poses, embeds their information into node hidden states,
  # performs message passing along the graph using node and edge modules,
  # and outputs an extracted embedding based on the position of queries
  def forward(self, view_frames, view_poses, query_poses, unsqueeze=True):
    bs = view_frames.shape[0]

    # Inputs to node hidden states
    nodes_states, more = self.inp_to_graph_inp(view_frames, view_poses)
    assert(nodes_states.shape == (bs, self.structure.num_nodes, self.structure.node_dim))

    # Pointing tensors
    edge_sources = torch.LongTensor(
        np.array(self.structure.edge_sources, dtype=int)).to(view_frames.device)
    edge_sinks = torch.LongTensor(
        np.array(self.structure.edge_sinks, dtype=int)).to(view_frames.device)
    
    # Perform message passing along edges
    for step in range(self.structure.msg_steps):
      sources = nodes_states[:, edge_sources, :].clone()
      sinks = nodes_states[:, edge_sinks, :].clone()

      inp = torch.cat([sources, sinks], dim=2)
      out = self.edge_module(inp.view(-1, inp.shape[2])).view(bs, -1, self.structure.msg_sz)

      incoming_msgs = torch.zeros(bs, self.structure.num_nodes, self.structure.msg_sz).to(view_frames.device)
      incoming_msgs = incoming_msgs.index_add(dim=1, index=edge_sinks, tensor=out)

      # Update node hidden states based on messages received
      # Every node shares the same node_module
      msgs_and_nodes_states = torch.cat([ incoming_msgs, nodes_states ], dim=2).view(
          -1, self.structure.node_dim + self.structure.msg_sz)
      update = self.node_module(msgs_and_nodes_states).view(bs, -1, self.structure.update_sz)
      nodes_states[:, :, -self.structure.update_sz:] += update

    # Return embeddings extracted from final node hidden states based on query positions
    embedding_out, more = self.graph_out_to_out(nodes_states, query_poses)

    # Tile num_poses into batch size and unsqueeze two tail dimensions
    if unsqueeze: embedding_out = embedding_out.unsqueeze(-1).unsqueeze(-1)

    return embedding_out, more
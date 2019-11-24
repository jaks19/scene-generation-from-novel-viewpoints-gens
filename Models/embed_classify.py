import torch
from torch import nn
from torch.autograd import Variable 
from torch.functional import F
from GEN.composer import GEN_Composer
from GQN.composer import GQN_Composer
import time

import Utils.global_vars as glo


class Classify(nn.Module):
    def __init__(self, baseline=False):
        super(Classify, self).__init__()

        if not baseline: 
            self.number_of_coordinates_copies = 8
            self.composer = GEN_Composer(num_copies=self.number_of_coordinates_copies)
            self.post_embedding_processor = PostEmbeddingProcessorGEN(num_copies=self.number_of_coordinates_copies)
        else: 
            self.number_of_coordinates_copies = 32
            self.composer = GQN_Composer(num_copies=self.number_of_coordinates_copies)
            self.post_embedding_processor = PostEmbeddingProcessorBaseline(num_copies=self.number_of_coordinates_copies)
        
        self.scalar = nn.Parameter(torch.FloatTensor([0]), requires_grad=True)
        self.softmax = nn.Softmax(dim=1)

        self.candidates_encoder = CandidatesEncoder()  
        return

    def forward(self, x, v, v_q, candidates, writer=None, i=None):
        assert(v_q.shape[0]==x.shape[0])

        # Obtain a bs x node_state_dim embedding for each query pose
        embedded_query_frames = self.composer(x, v, v_q, unsqueeze=False)
        embedded_query_frames = embedded_query_frames.view(-1, embedded_query_frames.shape[2])
        assert(embedded_query_frames.shape == (v_q.shape[0]*v_q.shape[1], 256 + (self.number_of_coordinates_copies*7))) 

        if writer:
            writer.add_scalar('mean of norm of raw embedding with concat coords (train)', torch.mean(torch.norm(embedded_query_frames, dim=1, keepdim=True)), i)
            writer.add_scalar('std of norm of raw embedding with concat coords (train)', torch.std(torch.norm(embedded_query_frames, dim=1, keepdim=True)), i)
            writer.add_scalar('var of norm of raw embedding with concat coords (train)', torch.var(torch.norm(embedded_query_frames, dim=1, keepdim=True)), i)

        # Feed to 2-layer feed forward to get to size 254
        embedded_query_frames = self.post_embedding_processor(embedded_query_frames)
        assert(embedded_query_frames.shape == (v_q.shape[0]*v_q.shape[1], 254))

        # Convolutional encoding of candidates without the composer, to size 254
        candidates_embeddings = self.candidates_encoder(candidates)
        assert(candidates_embeddings.shape == (candidates.shape[0], 254))

        return embedded_query_frames, candidates_embeddings


class CandidatesEncoder(nn.Module):
    def __init__(self):
        super(CandidatesEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(256, 254, kernel_size=1, stride=1)
        self.pool  = nn.AvgPool2d(16)

    def forward(self, x):
        skip_in  = F.relu(self.conv1(x))
        skip_out = F.relu(self.conv2(skip_in))

        r = F.relu(self.conv3(skip_in))
        r = F.relu(self.conv4(r)) + skip_out
        
        skip_out  = F.relu(self.conv5(r))

        r = F.relu(self.conv6(r))
        r = F.relu(self.conv7(r)) + skip_out
        r = F.relu(self.conv8(r))
        r = self.pool(r).squeeze(3).squeeze(2)
        return r


class PostEmbeddingProcessorGEN(nn.Module):
  def __init__(self, num_copies=None):
    super(PostEmbeddingProcessorGEN, self).__init__()
    in_sz = 256 + (num_copies*7)
    self.fc0 = nn.Linear(in_features=in_sz, out_features=256)
    self.fc1 = nn.Linear(in_features=256, out_features=254)

  def forward(self, x):
    x = F.relu(self.fc0(x))
    return self.fc1(x)


class PostEmbeddingProcessorBaseline(nn.Module):
  def __init__(self, num_copies=None):
    super(PostEmbeddingProcessorBaseline, self).__init__()
    in_sz = 256 + (num_copies*7)
    self.fc0 = nn.Linear(in_features=in_sz, out_features=512)
    self.fc1 = nn.Linear(in_features=512, out_features=512)
    self.fc2 = nn.Linear(in_features=512, out_features=512)
    self.fc3 = nn.Linear(in_features=512, out_features=254)

  def forward(self, x):
    x = F.relu(self.fc0(x))
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return self.fc3(x)


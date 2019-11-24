import collections, os, io
from PIL import Image
import torch
from torchvision.transforms import ToTensor, Resize
from torch.utils.data import Dataset
import random
import math as m
import numpy as np
import Utils.global_vars as glo

Context = collections.namedtuple('Context', ['frames', 'cameras'])
Scene = collections.namedtuple('Scene', ['frames', 'cameras'])

class Dataset_Custom(Dataset):
    def __init__(self, root_dir, force_size=None, allow_multiple_passes=False, num_main_coords=0):
        self.root_dir = root_dir
        self.size = force_size if force_size != None else len(os.listdir(self.root_dir))
        self.allow_multiple_passes = allow_multiple_passes
        self.need_main_coords = (num_main_coords > 0)
        self.number_main_coords = num_main_coords

    def __len__(self):
        return self.size if not self.allow_multiple_passes else self.size**2

    def __getitem__(self, idx):
        if self.allow_multiple_passes: idx = idx % self.size
        scene_path = os.path.join(self.root_dir, "{}.pt".format(idx))
        data = torch.load(scene_path)

        byte_to_tensor = lambda x: ToTensor()(Resize(glo.IMG_SIZE)((Image.open(io.BytesIO(x)))))
        images = torch.stack([byte_to_tensor(frame) for frame in data.frames])
        raw_viewpoints = torch.from_numpy(data.cameras).view(-1, 5)
        viewpoints = self.transform_viewpoint(raw_viewpoints)

        if self.need_main_coords:
            main_coords = self.get_main_coords(self.number_main_coords, raw_viewpoints)
            return images, viewpoints, main_coords        
        
        return images, viewpoints

    def transform_viewpoint(self, v):
        # Originally, v comes as 3D position, [yaw, pitch]
        w, z = torch.split(v, 3, dim=-1)
        y, p = torch.split(z, 1, dim=-1)
        view_vector = [w, torch.cos(y), torch.sin(y), torch.cos(p), torch.sin(p)]
        v_hat = torch.cat(view_vector, dim=-1)
        return v_hat

    def get_main_coords(self, num_main_points, coords_tensor):
        # Count number of images snapped from each coordinate in untouched dataset
        dic = {}
        for coord in coords_tensor:
            xy = str(coord[0].item())+','+str(coord[1].item())
            if xy in dic: dic[xy] += 1
            else: dic[xy] = 1
                
        s_dic = sorted(dic, key=lambda k: dic[k])
        s_dic.reverse()

        # Return coordinates of those points (from head of list)
        return [s_dic[i] for i in range(num_main_points)]

    
def sample_batch(x_data=None, v_data=None, D=None, expected_bs=None, scenes_per_dim=1, \
    shift=(0.0,0.0), seed=None, need_candidates=False, force_candidates=False, M=False, shuffle=True):

    if seed: random.seed(seed)

    # Cut off extra scenes along fake batch dimension 
    correct_fake_bs = expected_bs * scenes_per_dim**2
    x_data, v_data = x_data[:correct_fake_bs], v_data[:correct_fake_bs]
    x_data, v_data = organize_hotel_data(x_data, v_data, expected_bs, scenes_per_dim, shift)
    scenes_per_hotel = scenes_per_dim**2

    # N=Total available views per scene, K=how many to take to get embedding?
    if M:
        if D == "Room": N, V, Q = 10, M, 1
        elif D == "Labyrinth": N, V, Q = 300, M, 1

    else:
        if D == "Room": N, V, Q = 10, 5, 1
        elif D == "Labyrinth": N, V, Q = 300, 8, 1

    if shuffle:
        context_idx = torch.LongTensor([[random.sample(range(N*i, N*(i+1)), V)] for i in range(scenes_per_hotel)]).reshape(-1)
        query_idx = torch.LongTensor([[random.sample(range(N*i, N*(i+1)), Q)] for i in range(scenes_per_hotel)]).reshape(-1)
    
    else:
        context_idx = torch.LongTensor([[[j for j in range(i*V, (i+1)*V, 1)]] for i in range(scenes_per_hotel)]).reshape(-1)
        query_idx = torch.LongTensor([[[j for j in range(i*Q, (i+1)*Q, 1)]] for i in range(scenes_per_hotel)]).reshape(-1)
    
    x, v = x_data[:, context_idx], v_data[:, context_idx]
    x_q, v_q = x_data[:, query_idx], v_data[:, query_idx]

    if force_candidates: 
        context_idx = torch.LongTensor([[random.sample(range(N*i, N*(i+1)), V)] for i in range(scenes_per_hotel)]).reshape(-1)
        x_forced = x_data[:, context_idx]

    # Combine all query frames (at the head) and view frames (tail) for the full candidates bucket
    if need_candidates:
        if not(force_candidates):
            candidates_bucket = torch.cat([x_q.view(-1, 3, glo.IMG_SIZE, glo.IMG_SIZE), x.view(-1, 3, glo.IMG_SIZE, glo.IMG_SIZE)])
            assert(candidates_bucket.shape == ((x.shape[0]*(x.shape[1]+x_q.shape[1])), 3, glo.IMG_SIZE, glo.IMG_SIZE))

        else: candidates_bucket = torch.cat([x_q.view(-1, 3, glo.IMG_SIZE, glo.IMG_SIZE), x_forced.view(-1, 3, glo.IMG_SIZE, glo.IMG_SIZE)])   
        
        answer_indices = [i for i in range(x_q.shape[0]*x_q.shape[1])]
        return x, v, x_q, v_q, candidates_bucket, answer_indices

    print('Done')
    return x, v, x_q, v_q


def organize_hotel_data(x_data, v_data, expected_bs, scenes_per_dim, shift):
    bs_received, views = x_data.shape[:2]

    # Hotels params
    assert(int(bs_received / expected_bs) == bs_received / expected_bs)
    scenes_per_hotel = int(bs_received / expected_bs)
    assert(m.sqrt(scenes_per_hotel) == scenes_per_dim)

    # Data into hotels with the batch size expected by program
    # (batch 1: (room 1: (views 1-v)) (room2: (views 1-v))...) (batch 2: (room 1: (views 1-v)) (room2: (views 1-v))...)...
    x_data = x_data.view(expected_bs, views*scenes_per_hotel, 3, glo.IMG_SIZE, glo.IMG_SIZE)
    v_data = v_data.view(expected_bs, views*scenes_per_hotel, 7)

    # Correction to poses
    # 1. Shift poses to make up one hotel
    # Have to make tensor be on cpu for np.repeat
    x_corr = np.repeat(torch.arange(0, scenes_per_dim, 1).unsqueeze(0).cpu(), repeats=scenes_per_dim*views).to(v_data.device)
    y_corr = np.repeat(np.repeat(torch.arange(0, scenes_per_dim, 1).unsqueeze(0).cpu(), repeats=views).unsqueeze(0), repeats=scenes_per_dim, axis=0).view(scenes_per_hotel*views).to(v_data.device)
    xy_corr = torch.cat((x_corr.unsqueeze(1), y_corr.unsqueeze(1)), dim=1)
    xy_corr_per_batch = torch.cat((xy_corr.float(), v_data.new_zeros(scenes_per_hotel*views, 5)), dim=1)
    xy_corr_all = xy_corr_per_batch.unsqueeze(0).repeat(expected_bs, 1, 1) * 2 # Factor of 2 as images are width 2. height 2.

    # 2. Shift hotel within a grid of 10x10 (recall that we start at the corner (-1,-1) and that a hotel is square)
    shift_tensor = v_data.new_zeros(xy_corr_all.shape[1:])
    shift_tensor[:,0]=shift[0]
    shift_tensor[:,1]=shift[1]
    shift_tensor = shift_tensor.unsqueeze(0).repeat(expected_bs, 1,1)

    # Combine shifts
    v_data += (xy_corr_all + shift_tensor)

    return x_data, v_data

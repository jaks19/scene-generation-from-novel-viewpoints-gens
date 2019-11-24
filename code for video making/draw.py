import argparse
import time
import collections, os, io
from PIL import Image
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
from Utils.dataset import Dataset_Custom, sample_batch, Scene
from Models.scene_render import Renderer
import random 
import numpy as np
import Utils.global_vars as glo
import math as m
from torchvision.transforms import ToTensor, Resize
import torchvision
from tqdm import tqdm
import networkx as networkx
import matplotlib.pyplot as plt

class DrawDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
    def __len__(self):
        return len(os.listdir(self.root_dir))
    
    def __getitem__(self, idx):
        scene_path = os.path.join(self.root_dir, "{}.pt".format(idx))
        data = None
        
        # Unzip and read
        scene_path = os.path.join(self.root_dir, "{}.pt".format(idx))
        data = torch.load(scene_path)
        
        # Convert frames to one images tensor
        byte_to_tensor = lambda x: ToTensor()(Resize(64)((Image.open(io.BytesIO(x)))))
        images = torch.stack([byte_to_tensor(frame) for frame in data.frames])

        # Convert poses to one viewpoints tensor
        unprocessed_viewpoints = torch.from_numpy(data.cameras).view(-1, 5)
        main_coords = self.get_main_coords(unprocessed_viewpoints)     


        viewpoints = self.transform_viewpoint(unprocessed_viewpoints)
        return images, viewpoints, main_coords


    def transform_viewpoint(self, v):
        w, z = torch.split(v, 3, dim=-1)
        y, p = torch.split(z, 1, dim=-1)

        # position, [yaw, pitch]
        view_vector = [w, torch.cos(y), torch.sin(y), torch.cos(p), torch.sin(p)]
        v_hat = torch.cat(view_vector, dim=-1)
        return v_hat


# Takes a 2D grid and returns coordinates in networkx reference frame forming an S-shaped path
# Assumes 1 position per maze
def grid_to_s(grid):
    n = len(grid)
    s_coords = []
    for x in range(n):
        x_true=x
        for y in range(n):
            y_true=y
            if x%2!=0: y_true = n-1-y
            s_coords.append(grid[x_true][y_true])
    return s_coords


# Takes a list of N main coordinates (x, y)
# Generates N*num_intervals x 3 tensor of (x, y, theta) for angle_start < theta < angle_end
# Note: -pi < any_angle < pi
# angle_start to angle_end traversed in increments of delta
def get_query_coords(main_coords_in_s_order, angle_start, angle_end, delta):
    v_q_not_tensor_yet = []
    for coord in main_coords_in_s_order:
        v_q_not_tensor_yet+=([torch.FloatTensor([coord[0], coord[1], 0, m.cos(angle), m.sin(angle), 1, 0]) for angle in np.arange(angle_start, angle_end, delta)])

    v_q = torch.stack(v_q_not_tensor_yet, dim=0)
    return v_q

# One query generation strategy
# Assumes one main_coord per maze
def teleport_along_s_shape(main_coords, num_main_coords, hotel_dim):
    assert num_main_coords == 1 and len(main_coords)==hotel_dim**2
    for i in range(hotel_dim): assert len(main_coords[i])==1
    main_coords_1D = [main_coords[i][0] for i in range(hotel_dim**2)]
    
    # This part is why cannot use with >1 main position from same maze
    # Shift every coordinate and put inside of a 2D array 
    # (so can reference using netqorksx grid coordinates i.e. (0,0)-based)
    main_coords_with_shifts = [[0 for i in range(hotel_dim)] for j in range(hotel_dim)] 
    for i in range(hotel_dim):
        for j in range(hotel_dim):
            shift = (i*2.0, j*2.0)
            main_coord = main_coords_1D[(i*hotel_dim)+j].split(',')
            main_coords_with_shifts[i][j] = [shift[0]+float(main_coord[0]), shift[1]+float(main_coord[1])]
    
    s_shape_coords_lst = grid_to_s(main_coords_with_shifts)    

    # Will turn head from -pi to pi at each standing position
    start = -3.13
    end = 3.13
    delta = abs(end-start) / abs(args.fps*args.spc)
    rotational_query_coords = get_query_coords(s_shape_coords_lst, start, end, delta)
    return rotational_query_coords

def teleport_along_s_shape_multi(main_coords, num_main_coords, hotel_dim):
    assert len(main_coords)==hotel_dim**2
    chunk_size = len(main_coords[0])
    for i in range(len(main_coords)): assert len(main_coords[i]) == chunk_size
    
    # Will turn head from -pi to pi at each standing position
    start = -3.13
    end = 3.13
    delta = abs(end-start) / abs(args.fps*args.spc)
    
    overall = []
    for maze_ind in range(hotel_dim**2):
        overall_maze = []
        chunk = main_coords[maze_ind]

        for c in chunk:
            coord = [float(ch) for ch in c.split(',')]
            all_angles = torch. cat([torch.FloatTensor([coord[0], coord[1], 0, m.cos(angle), m.sin(angle), 1, 0]) for angle in np.arange(start, end, delta)], dim=0)
            overall_maze.append(all_angles)
        overall += overall_maze
    
    final = torch.stack(overall, dim=0)
    try: final = final.reshape((hotel_dim**2, chunk_size*(int((abs(start-end)/delta))+1), 7))
    except: final = final.reshape((hotel_dim**2, chunk_size*(int((abs(start-end)/delta))), 7))
    
    assert(final.shape[0] == hotel_dim**2)
    assert(final.shape[1] % chunk_size == 0)
    assert(final.shape[2] == 7)

    _, v_q, _, _ = sample_batch(x_data=torch.rand(final.shape[0], final.shape[1], 3, 64, 64), v_data=final, D='Labyrinth', expected_bs=1, scenes_per_dim=hotel_dim, shift=(0,0), M=final.shape[1], shuffle=False)
    return v_q.squeeze(0)


from sklearn.neighbors import KNeighborsRegressor

# def get_rgbs(X,y, queries):
#     neigh = KNeighborsRegressor(n_neighbors=2)
#     neigh.fit(X,y)
#     return neigh.predict(queries)

def custom_distance(a,b):
    return (1e5* np.sum(np.abs(np.floor(a/2)-np.floor(b/2))) +
            np.linalg.norm(a-b))

def get_rgbs(X,y, queries):
    neigh = KNeighborsRegressor(n_neighbors=2, metric=custom_distance)
    neigh.fit(X,y)
    return neigh.predict(queries)

    
def do_plot(plt, X,y, hotel_dim):
    delta = 0.05
    min_x = -1 ; max_x = hotel_dim*2.0+min_x+delta
    ori_x = np.arange(min_x, max_x, delta)

    # min_x = -3 ; max_x = 3
    # ori_x = np.arange(min_x, max_x, 0.05)

    ori_y = ori_x
    q_x, q_y = np.meshgrid(ori_x, ori_y)
    q_x = q_x.reshape(-1,1)
    q_y = q_y.reshape(-1,1)
    queries = np.concatenate([q_x,q_y], axis=1)
    
    aug_X, aug_y = augment_data(X[:,:2], y, X[:,2:3])
    res = get_rgbs(aug_X, aug_y,queries)
    
    return q_x, q_y, res

def plot_funny_triangle(pose):    
    big = 0.21
    small = 0.2

    # x,y,theta
    A, theta = pose[:2].reshape(1,2), pose[2]
    # theta  = theta - np.pi/4 #theta to horizontal component, subtract 45d.
    a = A
    B = A + big * np.array([np.cos(theta), np.sin(theta)])
    b = A + small * np.array([np.cos(theta), np.sin(theta)])
    C = A + big * np.array([-np.sin(theta), np.cos(theta)])
    c = A + small * np.array([-np.sin(theta), np.cos(theta)])
    big_triangle = np.concatenate([B,A,C], axis=0)
    small_triangle = np.concatenate([b,a,c], axis=0)

    plt.scatter(big_triangle[1,0], big_triangle[1,1], c='k', s=10)
    plt.plot(big_triangle[:2,0], big_triangle[:2,1], c='k', lw=1)
    plt.plot(big_triangle[1:,0], big_triangle[1:,1], c='k', lw=1)
    t2 = plt.Polygon(small_triangle, color='yellow', alpha=0.7)
    plt.gca().add_patch(t2)


def augment_data(X, y, pose):
    pose = np.arccos(pose)
    dpose = np.pi/4*np.array([-1,-1/2,0,1/2,1])
    dr = 0.4*np.array([0,0.25,0.5,0.75,1])
    list_X = [X]
    list_y = [y]
    for i in range(5):
        for j in range(5):
            angle = dpose[j] + pose
            delta = dr[i]*np.concatenate([np.cos(angle),np.sin(angle)],1)
            list_X.append(X + delta)
            list_y.append(y)
    return np.concatenate(list_X, 0), np.concatenate(list_y,0)



''' Load the mentioned model '''
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Draw path')
    parser.add_argument('--model_path', type=str, help='path to your model')
    parser.add_argument('--data_dir', type=str, help='location of training data', default="/DockerMountPoint/data/mazes-torch/train-num")
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--device_ids', type=int, nargs='+', help='list of CUDA devices (default: [0,1,2,3])', default=[0])
    parser.add_argument('--hotel_dim', type=int, help='the number of mazes along one dimension (1 of 1,2,...,5)', default=3)
    parser.add_argument('--log_dir', type=str, help='location where to save images', default="/DockerMountPoint/images/")

    parser.add_argument('--seed', type=int, help='seed for randomness', default=3)
    parser.add_argument('--idx', nargs='+', type=int, help='list of forced ids', default=None)
    parser.add_argument('--fps', type=float, help='fps for video', default=2.1)
    parser.add_argument('--spc', type=float, help='seconds per (x,y) coordinate (how long to stay here == how finely to split the 360degrees)', default=5.0)
    parser.add_argument('--nmc', type=int, help='number main coords per maze', default=2)

    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)

    hotel_dim = args.hotel_dim
    loader_bs = 1 * hotel_dim**2
    model_path = args.model_path
    num_main_coords = args.nmc


    # Save the images of mazes generated to folder 'mazes;'
    if not os.path.exists(os.path.join(args.log_dir, 'mazes')): os.makedirs(os.path.join(args.log_dir, 'mazes'))
    # Save the images of grids to folder 'grids'
    if not os.path.exists(os.path.join(args.log_dir, 'grids')): os.makedirs(os.path.join(args.log_dir, 'grids'))
        
    # writer = SummaryWriter(log_dir=os.path.join(args.log_dir,'images')
    device = f"cuda:{args.device_ids[0]}" if torch.cuda.is_available() else "cpu"
    data_dir = args.data_dir
    dataset = Dataset_Custom(root_dir=data_dir, num_main_coords=num_main_coords)

    kwargs = {'num_workers':args.workers, 'pin_memory': True} if torch.cuda.is_available() else {}

    model = Renderer(L=8, baseline=False).to(device)
    model = nn.DataParallel(model, device_ids=args.device_ids)

    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.module.composer.refresh_structure(hotel_dim, shift=(0,0))

    if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
        
    ''' Grab 300 images (all of them) from hotem_dim**2 total mazes '''
    all_images = []
    all_viewpoints = []
    main_coords = []

    # By not using DataLoader, can provide own ids and also 
    # we return the main position(s) in each maze where most pictures were taken
    for i in range(hotel_dim**2):
        if args.idx: query_idx = args.idx[i]
        else: query_idx = random.randint(1,10000)
        images, viewpoints, main_coords_list = dataset[query_idx]
        all_images.append(images)
        all_viewpoints.append(viewpoints)
        main_coords.append(main_coords_list)

    # All views from all mazes which make up the hotel
    x_data = torch.stack(all_images).to(device)
    v_data = torch.stack(all_viewpoints).to(device)

    assert(x_data.shape == (hotel_dim**2, 300, 3, 64, 64))
    assert(v_data.shape == (hotel_dim**2, 300, 7))

    # Data: inputs for building embedding for hotel 
    x, v, _, _ = sample_batch(x_data=x_data.clone(), v_data=v_data.clone(), D='Labyrinth', expected_bs=1, scenes_per_dim=hotel_dim, M=8)
    assert(x.shape==(1, 8*hotel_dim**2, 3, 64, 64) and v.shape==(1, 8*hotel_dim**2, 7))

    # Data: to build the color prior for each maze (all 300 images) 
    # Use sample_batch to shift poses to make hotel (no sub-sampling due to M=300)
    x_data, v_data, _, _ = sample_batch(x_data=x_data.clone(), v_data=v_data.clone(), D='Labyrinth', expected_bs=1, scenes_per_dim=hotel_dim,M=300, shuffle=False)
    assert(x_data.shape==(1, 300*hotel_dim**2, 3, 64, 64) and v_data.shape==(1, 300*hotel_dim**2, 7))

    # Grab the data to scatter for the background once
    x_r = x_data[:,:,:,60,50].clone().cpu().reshape((hotel_dim**2 * 300,3)).numpy()
    v_r = torch.cat([v_data[:,:,:2].reshape((hotel_dim**2 * 300,2)), v_data[:,:,3:4].reshape([hotel_dim**2 * 300,1])], dim=1).clone().cpu().numpy()

    bkg = do_plot(plt, v_r, x_r, hotel_dim)

    # Data: main Deepmind-chosen coords (may be useful to generate new query positions for video)
    # Pass through sample_batch for shift
    assert(len(main_coords) == hotel_dim**2)
    for i in range(hotel_dim**2): assert(len(main_coords[i])==num_main_coords)
        
    # v_q = teleport_along_s_shape(main_coords, num_main_coords, hotel_dim)
    v_q = teleport_along_s_shape_multi(main_coords, num_main_coords, hotel_dim).to(device)

    # # Infer the image for each query view using model, and save
    res, activation_weights = model.module.generate(x.to(device), v.to(device), v_q.unsqueeze(0))
    for i in range(res.shape[0]): torchvision.utils.save_image(res[i], os.path.join(os.path.join(args.log_dir, 'mazes'), '{}.png'.format(i)))

    # Indices of nodes activated for each pose
    # interpolation_coords[0] has index top_left, [1] has top_right, [2] has ..., [3] has ...
    # interpolation is not origin-position-dependent, as will return indices of nodes in a flattened 1D array
    interpolation_coords = model.module.composer.structure.get_interpolation_coordinates(v_q.unsqueeze(1), force=True)

    # Map the axes from the GEN
    GEN_structure = model.module.composer.structure
    nx = GEN_structure.grid['n_x']
    node_positions = [(npos[0].item(), npos[1].item()) for npos in GEN_structure.node_positions]
    dx = dy = GEN_structure.grid['dx']

    old = -1
    corrector = 1
    situation = 1

    for i in tqdm(range(v_q.shape[0])):
        # Background
        plt.scatter(bkg[0],bkg[1],s=100,c=bkg[2][:,:3], lw=0)
        
        # Activated nodes 
        activated_nodes = [node_positions[interpolation_coords[j][i]] for j in range(4)]
        wts = [activation_weights[k][0][i].item()**0.5 for k in [2,3,0,1]]
        nrm = np.linalg.norm(wts)
        midpt = [np.dot(np.array([n[k] for n in activated_nodes]), np.array(wts))/nrm for k in [0,1]]
        # midpt = torch.mean(torch.FloatTensor(activated_nodes), dim=0).numpy()

        # Query angle (corrector not tracked in 'old', 'new')
        # Corrector needed as cos-1(x)=cos-1(-x) 
        new = np.arccos(v_q[i][3].item())
        if old != -1: 
            if new > old and situation == 1: 
                corrector *= -1
                situation = 2
            if new < old and situation == 2:
                corrector *= -1
                situation = 1
                
        old = new
        plot_funny_triangle(np.array([np.array([midpt[0]]), np.array([midpt[1]]), corrector*new]))
        
        # Grid with nx*nx nodes taken directly from GEN node representation
        G = networkx.Graph()  
        G.add_nodes_from(node_positions)
        pos = dict(zip(G, node_positions))
        
        # Draw non-activated nodes as uniformly white
        for node in G:
            if not(node in activated_nodes):
                networkx.draw_networkx_nodes(G,pos,
                   nodelist=[node],
                   node_color='w',
                   node_size=80,
                   alpha=0.3)
        
        # Activated nodes red with opacity prop. to activation weight**0.33
        networkx.draw_networkx_nodes(G,pos,
           nodelist=activated_nodes,
           node_color=['r'],
           node_size=[150],
           alpha=[max(0.2,w**0.5) for w in wts])
        
        # Add edges between adjacent nodes
        Grid = networkx.grid_2d_graph(nx,nx)
        networkx.draw(Grid, dict(zip(Grid, node_positions)), node_size=[], node_color=[], with_labels=False)
        
        # Make Origin in top left to agree with GEN coordinates 
        # !!Important!! -- Everything is based on this assumption of alignment
        plt.setp(plt.gca(), 'ylim', list(reversed(plt.getp(plt.gca(), 'ylim'))))
        plt.axis('equal')
        
        plt.savefig(os.path.join(os.path.join(args.log_dir, 'grids'), '{}.png'.format(i)))
        plt.clf()
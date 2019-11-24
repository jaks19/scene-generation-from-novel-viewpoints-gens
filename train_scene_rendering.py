import argparse
import time
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from Utils.dataset import Dataset_Custom, sample_batch, Scene
from Utils.scheduler import AnnealingStepLR
from Models.scene_render import Renderer
import random 
import numpy as np
import Utils.global_vars as glo
import math as m

baseline = False
force_candidates = False
heavy_log_interval = 100
log_interval = 500
save_interval = 1000
force_size_train = 107999
force_size_test = 11999

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generative Query Network with GEN Implementation')
    parser.add_argument('--dataset', type=str, default='Labyrinth', help='dataset (dafault: Shepard-Mtzler)')
    parser.add_argument('--train_data_dir', type=str, help='location of training data', \
                        default="/home/jaks19/mazes-torch/train")
    parser.add_argument('--test_data_dir', type=str, help='location of test data', \
                        default="/home/jaks19/mazes-torch/test")
    parser.add_argument('--root_log_dir', type=str, help='root location of log', default='/home/jaks19/logs/')
    parser.add_argument('--log_dir', type=str, help='log directory (default: GQN)', default='GQN')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=32)
    parser.add_argument('--device_ids', type=int, nargs='+', help='list of CUDA devices (default: [0,1,2,3])', default=[0,1,2,3,4,5,6,7])
    parser.add_argument('--layers', type=int, help='number of generative layers (default: 12)', default=8)
    parser.add_argument('--saved_model', type=str, help='path to model', default=None)
    args = parser.parse_args()

    log_dir = os.path.join(args.root_log_dir, args.log_dir)
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, 'models')): os.makedirs(os.path.join(log_dir, 'models'))
    if not os.path.exists(os.path.join(log_dir,'runs')): os.makedirs(os.path.join(log_dir,'runs'))
    writer = SummaryWriter(log_dir=os.path.join(log_dir,'runs'))

    seed = 3
    torch.manual_seed(seed)
    random.seed(seed)

    min_train_structure_dim = 1
    max_train_structure_dim = 2
    min_test_structure_dim = 1
    max_test_structure_dim = 5
    interval_alter_structure_train = 1
    train_structure_refresh_needed = True
    scenes_per_dim_train = None
    scenes_per_dim_test = None
    shift_train = (0.0, 0.0)
    shift_test = (0.0, 0.0)
    
    # Data
    D = args.dataset
    B = 16
    B_test = 1
    loader_bs = [None, None]
    loader_bs[0] = B * max_train_structure_dim**2
    loader_bs[1] = B_test * max_test_structure_dim**2 
    # For parallel model, want batch size to be divisible
    assert(loader_bs[0] >= len(args.device_ids) and loader_bs[1] >= len(args.device_ids))

    train_data_dir = args.train_data_dir
    test_data_dir = args.test_data_dir
    train_dataset = Dataset_Custom(root_dir=train_data_dir, force_size=force_size_train, allow_multiple_passes=False)
    test_dataset = Dataset_Custom(root_dir=test_data_dir, force_size=force_size_test, allow_multiple_passes=False)

    kwargs = {'num_workers':args.workers, 'pin_memory': True} if torch.cuda.is_available() else {}

    train_loader = DataLoader(train_dataset, batch_size=loader_bs[0], shuffle=True, drop_last=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=loader_bs[1], shuffle=True, drop_last=True, **kwargs)
    train_iter = iter(train_loader)
    test_iter = iter(test_loader)

    device = f"cuda:{args.device_ids[0]}" if torch.cuda.is_available() else "cpu"
    model = Renderer(L=args.layers, baseline=baseline).to(device)
    if len(args.device_ids)>1: model = nn.DataParallel(model, device_ids=args.device_ids)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-4, betas=(0.9, 0.999), eps=1e-08)
    scheduler = AnnealingStepLR(optimizer, mu_i=8e-4, mu_f=5e-6, n=1.6e6)

    restoring_epoch = 0
    if args.saved_model != None: 
        checkpoint = torch.load(args.saved_model)
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        restoring_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Training parameters
    sigma_i, sigma_f = 2.0, 0.7
    sigma = sigma_i

    # minimum_clip_wait = 1000
    # clip_limit = 200

    total_epochs = 10**5
    for i in tqdm(range(total_epochs)):
        t = i + restoring_epoch
        start = time.time()

        try: x_data, v_data = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x_data, v_data = next(train_iter)

        x_data = x_data.to(device)
        v_data = v_data.to(device)

        if t%interval_alter_structure_train==0 or train_structure_refresh_needed:
            scenes_per_dim_train = random.randint(min_train_structure_dim, max_train_structure_dim)
            space_left = 10.0 - (scenes_per_dim_train*2.0)
            shift_train = (random.uniform(0,space_left), random.uniform(0,space_left))
            if len(args.device_ids)>1: model.module.composer.refresh_structure(scenes_per_dim_train, shift_train)
            else: model.composer.refresh_structure(scenes_per_dim_train, shift_train)
            train_structure_refresh_needed = False

        adjusted_bs = m.floor((B * (max_train_structure_dim**2)) / (scenes_per_dim_train**2))
        x, v, x_q, v_q = sample_batch(x_data=x_data, v_data=v_data, D=D, expected_bs=adjusted_bs, scenes_per_dim=scenes_per_dim_train, shift=shift_train)

        model.train()
        elbo = model(x, v, v_q, x_q, sigma)

        writer.add_scalar(f'train loss {scenes_per_dim_train}x{scenes_per_dim_train}', -elbo.mean(), t)
        writer.add_scalar('train_loss agg', -elbo.mean(), t)
        (-elbo.mean()).backward()
        # if t > minimum_clip_wait: nn.utils.clip_grad_norm_(model.parameters(), clip_limit, norm_type=2)

        optimizer.step()
        scheduler.step()

        end = time.time()
        writer.add_scalar('time per iter', end-start, t)

        # Debug plots: Norms of module weights and gradients
        if t % heavy_log_interval == 0:
            writer.add_scalar('norm(module weights)', sum([np.linalg.norm(param.data.clone().cpu()) for param in model.parameters()]), t)
            writer.add_scalar('norm(gradients all params)', sum([np.linalg.norm(param.grad.clone().cpu()) for param in model.parameters()]), t)
            writer.add_scalar('learning rate', sum([param_group['lr'] for param_group in optimizer.param_groups]), t)
        
        optimizer.zero_grad()
        # Pixel-variance annealing
        sigma = max(sigma_f + (sigma_i - sigma_f)*(1 - t/(2e5)), sigma_f)

        
        with torch.no_grad():
            if t % log_interval == 0:
                model.eval()

                # Logs pertaining to train data
                if len(args.device_ids)>1:
                    kl_train = model.module.kl_divergence(x, v, v_q, x_q)
                    x_q_rec_train = model.module.reconstruct(x, v, v_q, x_q)
                    x_q_hat_train = model.module.generate(x, v, v_q)
                else:
                    kl_train = model.kl_divergence(x, v, v_q, x_q)
                    x_q_rec_train = model.reconstruct(x, v, v_q, x_q)
                    x_q_hat_train = model.generate(x, v, v_q)
                
                s = x_q.shape
                writer.add_scalar('train kl', kl_train.mean(), t)
                writer.add_image('train ground truth', make_grid(x_q.view(s[0]*s[1],3,glo.IMG_SIZE,glo.IMG_SIZE), 6, pad_value=1), t)
                writer.add_image('train reconstruction', make_grid(x_q_rec_train.view(s[0]*s[1],3,glo.IMG_SIZE,glo.IMG_SIZE), 6, pad_value=1), t)
                writer.add_image('train generation', make_grid(x_q_hat_train.view(s[0]*s[1],3,glo.IMG_SIZE,glo.IMG_SIZE), 6, pad_value=1), t)

                # Logs pertaining to test data
                try: x_data_test_raw, v_data_test_raw = next(test_iter)
                except StopIteration:
                    test_iter = iter(test_loader)
                    x_data_test_raw, v_data_test_raw = next(test_iter)

                for scenes_per_dim_test in range(min_test_structure_dim, max_test_structure_dim+1):
                    space_left = 10.0 - (scenes_per_dim_test*2.0)
                    shift_test = (random.uniform(0,space_left), random.uniform(0,space_left)) 
                    if len(args.device_ids)>1: model.module.composer.refresh_structure(scenes_per_dim_test, shift_test)
                    else: model.composer.refresh_structure(scenes_per_dim_test, shift_test)
                    
                    x_data_test = x_data_test_raw.clone().to(device)
                    v_data_test = v_data_test_raw.clone().to(device)
                    adjusted_bs = m.floor((B_test * (max_test_structure_dim**2)) / (scenes_per_dim_test**2))
                    x_test, v_test, x_q_test, v_q_test = sample_batch(x_data=x_data_test, v_data=v_data_test, D=D, expected_bs=adjusted_bs, scenes_per_dim=scenes_per_dim_test, shift=shift_test)
                    elbo_test = model(x_test, v_test, v_q_test, x_q_test, sigma)                

                    if len(args.device_ids)>1:
                        kl_test = model.module.kl_divergence(x_test, v_test, v_q_test, x_q_test)
                        x_q_rec_test = model.module.reconstruct(x_test, v_test, v_q_test, x_q_test)
                        x_q_hat_test = model.module.generate(x_test, v_test, v_q_test)
                    else:
                        kl_test = model.kl_divergence(x_test, v_test, v_q_test, x_q_test)
                        x_q_rec_test = model.reconstruct(x_test, v_test, v_q_test, x_q_test)
                        x_q_hat_test = model.generate(x_test, v_test, v_q_test)

                    s = x_q_test.shape
                    writer.add_scalar(f'test loss {scenes_per_dim_test}x{scenes_per_dim_test}', -elbo_test.mean(), t)
                    writer.add_scalar(f'test kl {scenes_per_dim_test}x{scenes_per_dim_test}', kl_test.mean(), t)
                    writer.add_image(f'test ground truth {scenes_per_dim_test}x{scenes_per_dim_test}', make_grid(x_q_test.view(s[0]*s[1],3,glo.IMG_SIZE,glo.IMG_SIZE), 6, pad_value=1), t)
                    writer.add_image(f'test reconstruction {scenes_per_dim_test}x{scenes_per_dim_test}', make_grid(x_q_rec_test.view(s[0]*s[1],3,glo.IMG_SIZE,glo.IMG_SIZE), 6, pad_value=1), t)
                    writer.add_image(f'test generation {scenes_per_dim_test}x{scenes_per_dim_test}', make_grid(x_q_hat_test.view(s[0]*s[1],3,glo.IMG_SIZE,glo.IMG_SIZE), 6, pad_value=1), t)
                
                # Reset to a train structure in next iter
                train_structure_refresh_needed = True

            if t % save_interval == 0:
                torch.save({
                    'epoch': t,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                    }, log_dir + "/models/checkpoint-{}.pt".format(t))

    torch.save({
        'epoch': total_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
        }, log_dir + "/models/checkpoint-final.pt")
    writer.close()


import torch
from torch import nn
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from GQN.modules import InferenceCore, GenerationCore, Conv2dLSTMCell
from GQN.composer import GQN_Composer
from GEN.composer import GEN_Composer
import time

import Utils.global_vars as glo

class Renderer(nn.Module):
    def __init__(self, L=8, baseline=False):
        super(Renderer, self).__init__()
        
        # Number of generative layers
        self.L = L
        
        # Composer
        self.baseline = baseline
        if not baseline: 
            self.number_of_coordinates_copies = 8
            self.composer = GEN_Composer(num_copies=self.number_of_coordinates_copies)
        else: 
            self.number_of_coordinates_copies = 32
            self.composer = GQN_Composer(num_copies=self.number_of_coordinates_copies)

        # Cores
        self.inference_core = InferenceCore(num_copies=self.number_of_coordinates_copies)
        self.generation_core = GenerationCore(num_copies=self.number_of_coordinates_copies)
            
        # Reusable embedders
        self.eta_pi = nn.Conv2d(128, 2*3, kernel_size=5, stride=1, padding=2)
        
        if glo.IMG_SIZE == 64: self.eta_g = nn.Conv2d(128, 3, kernel_size=1, stride=1, padding=0)
        elif glo.IMG_SIZE == 16: self.eta_g = nn.Conv2d(128, 3, kernel_size=1, stride=4, padding=0)
        else: raise NotImplementedError

        self.eta_e = nn.Conv2d(128, 2*3, kernel_size=5, stride=1, padding=2)

    # EstimateELBO
    def forward(self, x, v, v_q, x_q, sigma):
        m, n = v_q.shape[:2]
        M = x.shape[1]
        B = m*n

        # Obtain a B x node_state_dim embedding (one per query image) from GEN
        r = self.composer(x, v, v_q).view(B, 256+(7*self.number_of_coordinates_copies), 1, 1)

        v_q = v_q.view(B, 7)
        x_q = x_q.view(B, 3, glo.IMG_SIZE, glo.IMG_SIZE)

        # Generator initial state
        c_g = x.new_zeros((B, 128, 16, 16))
        h_g = x.new_zeros((B, 128, 16, 16))
        u = x.new_zeros((B, 128, 64, 64))

        # Inference initial state
        c_e = x.new_zeros((B, 128, 16, 16))
        h_e = x.new_zeros((B, 128, 16, 16))
                
        elbo = 0

        for l in range(self.L):

            # Prior factor
            mu_pi, logvar_pi = torch.split(self.eta_pi(h_g), 3, dim=1)
            std_pi = torch.exp(0.5*logvar_pi)
            pi = Normal(mu_pi, std_pi)
            
            # Inference state update
            c_e, h_e = self.inference_core(x_q, v_q, r, c_e, h_e, h_g, u)

            # Posterior factor
            mu_q, logvar_q = torch.split(self.eta_e(h_e), 3, dim=1)
            std_q = torch.exp(0.5*logvar_q)
            q = Normal(mu_q, std_q)
            
            # Posterior sample
            z = q.rsample()
            
            # Generator state update
            c_g, h_g, u = self.generation_core(v_q, r, c_g, h_g, u, z)

            # ELBO KL contribution update
            elbo -= torch.sum(kl_divergence(q, pi), dim=[1,2,3])

        # ELBO likelihood contribution update
        elbo += torch.sum(Normal(self.eta_g(u), sigma).log_prob(x_q), dim=[1,2,3])
        return elbo
    

    def generate(self, x, v, v_q):
        
        m, n = v_q.shape[:2]
        M = x.shape[1]
        B = m*n
        
        # Obtain a B x node_state_dim embedding (one per scene) from GEN
        r, more = self.composer(x, v, v_q)
        r = r.view(B, 256+(7*self.number_of_coordinates_copies), 1, 1)

        v_q = v_q.view(B, 7)

        # Initial state
        c_g = x.new_zeros((B, 128, 16, 16))
        h_g = x.new_zeros((B, 128, 16, 16))
        u = x.new_zeros((B, 128, 64, 64))
        
        for l in range(self.L):
            # Prior factor
            mu_pi, logvar_pi = torch.split(self.eta_pi(h_g), 3, dim=1)
            std_pi = torch.exp(0.5*logvar_pi)
            pi = Normal(mu_pi, std_pi)
            
            # Prior sample
            z = pi.sample()
            
            # State update
            c_g, h_g, u = self.generation_core(v_q, r, c_g, h_g, u, z)
            
        # Image sample
        mu = self.eta_g(u)

        return torch.clamp(mu, 0, 1), more
    
    def kl_divergence(self, x, v, v_q, x_q):
        
        m, n = v_q.shape[:2]
        M = x.shape[1]
        B = m*n
        
        # Obtain a B x node_state_dim embedding (one per scene) from GEN
        r = self.composer(x, v, v_q).view(B, 256+(7*self.number_of_coordinates_copies), 1, 1)

        v_q = v_q.view(B, 7)
        x_q = x_q.view(B, 3, glo.IMG_SIZE, glo.IMG_SIZE)
            
        # Generator initial state
        c_g = x.new_zeros((B, 128, 16, 16))
        h_g = x.new_zeros((B, 128, 16, 16))
        u = x.new_zeros((B, 128, 64, 64))

        # Inference initial state
        c_e = x.new_zeros((B, 128, 16, 16))
        h_e = x.new_zeros((B, 128, 16, 16))
                
        kl = 0
        for l in range(self.L):
            # Prior factor
            mu_pi, logvar_pi = torch.split(self.eta_pi(h_g), 3, dim=1)
            std_pi = torch.exp(0.5*logvar_pi)
            pi = Normal(mu_pi, std_pi)
            
            c_e, h_e = self.inference_core(x_q, v_q, r, c_e, h_e, h_g, u)
            
            # Posterior factor
            mu_q, logvar_q = torch.split(self.eta_e(h_e), 3, dim=1)
            std_q = torch.exp(0.5*logvar_q)
            q = Normal(mu_q, std_q)
            
            # Posterior sample
            z = q.rsample()
            
            # Generator state update
            c_g, h_g, u = self.generation_core(v_q, r, c_g, h_g, u, z)
                
            # ELBO KL contribution update
            kl += torch.sum(kl_divergence(q, pi), dim=[1,2,3])

        return kl
    
    def reconstruct(self, x, v, v_q, x_q):
        
        m, n = v_q.shape[:2]
        M = x.shape[1]
        B = m*n
        
        # Obtain a B x node_state_dim embedding (one per scene) from GEN
        r = self.composer(x, v, v_q).view(B, 256+(7*self.number_of_coordinates_copies), 1, 1)

        v_q = v_q.view(B, 7)
        x_q = x_q.view(B, 3, glo.IMG_SIZE, glo.IMG_SIZE)
            
        # Generator initial state
        c_g = x.new_zeros((B, 128, 16, 16))
        h_g = x.new_zeros((B, 128, 16, 16))
        u = x.new_zeros((B, 128, 64, 64))

        # Inference initial state
        c_e = x.new_zeros((B, 128, 16, 16))
        h_e = x.new_zeros((B, 128, 16, 16))
                
        for l in range(self.L):
            # Inference state update
            c_e, h_e = self.inference_core(x_q, v_q, r, c_e, h_e, h_g, u)
            
            # Posterior factor
            mu_q, logvar_q = torch.split(self.eta_e(h_e), 3, dim=1)
            std_q = torch.exp(0.5*logvar_q)
            q = Normal(mu_q, std_q)
            
            # Posterior sample
            z = q.rsample()
            
            # Generator state update
            c_g, h_g, u = self.generation_core(v_q, r, c_g, h_g, u, z)

        mu = self.eta_g(u)
        return torch.clamp(mu, 0, 1)
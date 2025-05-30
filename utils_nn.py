import os
import glob
import typing as t
from copy import deepcopy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from timm import create_model

from utils import au

import matplotlib.pyplot as plt


class CustomDataset(Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
        self.no_samples = len(glob.glob1(self.data_dir, "*.npy"))
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize((224, 224)),
                                             transforms.Normalize(mean=[0.5], std=[0.5])])
        
    def __len__(self):
        return self.no_samples

    def __getitem__(self, idx):
        C_map = np.load(os.path.join(self.data_dir, 'sample_'+str(idx)+'.npy'))
        C_map0 = torch.tensor(C_map, dtype=torch.float32)  # 101x101
        C_map = torch.stack([self.transform(np.array(C_m)) for C_m in C_map]).squeeze(1).to(torch.float32)  # 224x224 for ViT
        label = np.loadtxt(os.path.join(self.data_dir, 'label_'+str(idx)+'.txt'))
        label[:5] *= au.Eh  # to be in meVs
        return C_map, C_map0, torch.tensor(label).type((torch.float32))

def parse_dataset(trainset_directory, testset_directory, batch_size=100):
    train_loader = DataLoader(CustomDataset(trainset_directory), batch_size=batch_size)
    test_loader   = DataLoader(CustomDataset(testset_directory), batch_size=batch_size)
    return train_loader, test_loader


class Hamiltonian:
    
    def __init__(self, parameters, device):
        self.p = parameters
        self.d = device
        self.sx = torch.tensor([[0,1],[1,0]], dtype=torch.complex64, device=device)
        self.sy = torch.tensor([[0,-1j],[1j,0]], dtype=torch.complex64, device=device)
        self.sz = torch.tensor([[1,0],[0,-1]], dtype=torch.complex64, device=device)
        
    def onsite_matrix(self, mu, i=0):
        par = self.p
        h_onsite = torch.zeros((mu.size()[0], 4, 4), dtype=torch.complex64, device=self.d)     
        h_onsite[:,0,0] = -mu+par.b[i]*au.Eh 
        h_onsite[:,1,1] = -mu-par.b[i]*au.Eh 
        h_onsite[:,2,2] =  mu-par.b[i]*au.Eh 
        h_onsite[:,3,3] =  mu+par.b[i]*au.Eh 
        h_onsite[:,0,3] =  par.d[i]*au.Eh*torch.exp(torch.tensor( 1.j)*par.ph_d*i)
        h_onsite[:,1,2] = -par.d[i]*au.Eh*torch.exp(torch.tensor( 1.j)*par.ph_d*i)
        h_onsite[:,2,1] = -par.d[i]*au.Eh*torch.exp(torch.tensor(-1.j)*par.ph_d*i)
        h_onsite[:,3,0] =  par.d[i]*au.Eh*torch.exp(torch.tensor(-1.j)*par.ph_d*i)
        return h_onsite
    
    def hopping_matrix(self, t, l, i=0):
        par = self.p
        hopping = torch.zeros((t.size()[0], 4, 4), dtype=torch.complex64, device=self.d)
        phr = torch.tensordot((t*torch.cos(l)).to(torch.float32), torch.eye(2, device=self.d), dims=0) 
        l_rho = torch.tensor(par.l_rho[i], dtype=torch.float32, device=self.d)
        l_ksi = torch.tensor(par.l_ksi[i], dtype=torch.float32, device=self.d)
        lambda_versor = [torch.sin(l_rho)*torch.cos(l_ksi), torch.sin(l_rho)*torch.sin(l_ksi), torch.cos(l_rho)]
        phi = torch.tensordot((t*torch.sin(l)).to(torch.complex64), self.sx*lambda_versor[0]+self.sy*lambda_versor[1]+self.sz*lambda_versor[2], dims=0)
        hopping[:,0:2,0:2] = phi*1j + phr
        phi = torch.tensordot((t*torch.sin(l)).to(torch.complex64), self.sx*lambda_versor[0]+self.sy.T*lambda_versor[1]+self.sz*lambda_versor[2], dims=0)
        hopping[:,2:4,2:4] = phi*1j - phr
        return hopping
   
    def generate(self, predicted_params):
        # predicted_params: batch x [mu_L, mu_C, mu_R, t_LC, t_CR, l_LC, l_CR]
        # everything in meVs !
        h_tensor = torch.zeros((predicted_params.size(0), 12, 12), dtype=torch.complex64, device=self.d)
        h_tensor[:,0:4 ,0:4 ] = self.onsite_matrix(predicted_params[:,0])
        h_tensor[:,4:8 ,4:8 ] = self.onsite_matrix(predicted_params[:,1])
        h_tensor[:,8:12,8:12] = self.onsite_matrix(predicted_params[:,2])
        h_tensor[:,0:4 ,4:8 ] = self.hopping_matrix(predicted_params[:,3], predicted_params[:,5])
        h_tensor[:,4:8 ,8:12] = self.hopping_matrix(predicted_params[:,4], predicted_params[:,6])
        h_tensor[:,4:8 ,0:4 ] = torch.conj(self.hopping_matrix(predicted_params[:,3], predicted_params[:,5]).transpose(1,2))
        h_tensor[:,8:12,4:8 ] = torch.conj(self.hopping_matrix(predicted_params[:,4], predicted_params[:,6]).transpose(1,2))
        return h_tensor
    

class Transport:
    def __init__(self, reference_point, device):
        self.ref_mu = reference_point.mu*au.Eh 
        self.ref_b = reference_point.b*au.Eh 
        self.d = device

    def torch_conductance_collection(self, h_tensor, predicted_params):
        C_map = []
        for k in [0,1,2]:
            for i in [0,1]:
                for j in [0,1]:
                    C_map.append(self.torch_conductance_map21(h_tensor, predicted_params, i, j, k))
        for i in [0,1]:
            for j in [0,1]:
                C_map.append(self.torch_conductance_map22(h_tensor, i, j))
        return torch.stack(C_map, axis=-3)
    
    def torch_conductance_map21(self,
        h_tensor: torch.Tensor, # must be complex and in standard representation
        predicted_params: torch.Tensor,
        i: float,
        j: float,
        k: float,
        ef_range: t.Tuple[float, float] = (-1.5, 1.5),
        det_range: t.Tuple[float, float] = (-1., 1.),
        ef_num: int = 101,
        det_num: int = 101,
        gamma: float = 0.1
    ) -> torch.Tensor:
        efs = torch.linspace(ef_range[0], ef_range[1], steps=ef_num).to(h_tensor.device)
        dets = torch.linspace(det_range[0], det_range[1], steps=det_num).to(h_tensor.device)
        # detune k-dot only, self.ref_mu is a reference
        mus_expanded = torch.zeros((det_num, 3), device=self.d)
        mus_expanded[:, k] = dets
        mus_expanded = mus_expanded.view(det_num, 1, 3).expand(det_num, h_tensor.shape[0], 3).clone()
        mus_expanded += predicted_params[:,:3].expand(det_num, h_tensor.shape[0], 3)
        hs = h_tensor.view(1, *h_tensor.shape).expand(det_num, *h_tensor.shape)
        hs_modified = self.torch_h_set_mu(hs, mus_expanded)
        cmap = self._torch_cmap(hs_modified, efs, i, j, 1, gamma)
        return cmap
    
    def torch_conductance_map22(self,
        h_tensor: torch.Tensor, # must be complex and in standard representation
        i: float,
        j: float,
        ef_range: t.Tuple[float, float] = (-1.5, 1.5),
        b_range: t.Tuple[float, float] = (-.5, .5),
        ef_num: int = 101,
        b_num: int = 101,
        gamma: float = 0.1
    ) -> torch.Tensor:

        efs = torch.linspace(ef_range[0], ef_range[1], steps=ef_num).to(h_tensor.device)
        bs = torch.linspace(b_range[0]+torch.tensor(self.ref_b[0]), b_range[1]+torch.tensor(self.ref_b[0]), steps=b_num).to(h_tensor.device)
        bs_expanded = bs.view(b_num, 1, 1).expand(b_num, *h_tensor.shape[:-2], 1).to(h_tensor.device)
        hs = h_tensor.view(1, *h_tensor.shape).expand(b_num, *h_tensor.shape)
        hs_modified = self.torch_h_set_b(hs, bs_expanded)
        cmap = self._torch_cmap(hs_modified, efs, i, j, 1, gamma)
        return cmap

    def _torch_cmap(self,
        h_tensor: torch.Tensor,
        efs: torch.Tensor, 
        i: float,
        j: float,
        n_levels: int = 1,
        gamma: float = 0.1
    ) -> torch.Tensor:

        w_matrix = self.torch_w_matrix(h_tensor.shape[-1], gamma, n_levels)
        s_matrix = self.torch_s_matrix(h_tensor, efs.view(efs.shape[0], *((1,)*len(h_tensor.shape))), w_matrix)

        dij = torch.eye(2)[i,j]
        n_dots = h_tensor.shape[-1] // (4 * n_levels)
        i *= n_dots - 1
        j *= n_dots - 1

        Sij = s_matrix.reshape(*s_matrix.shape[:-2], s_matrix.shape[-1] // 4, 2, 2, s_matrix.shape[-1] // 4, 2, 2)
        r_ee = Sij[..., i, 0, :, j, 0, :]
        r_he = Sij[..., i, 1, :, j, 0, :]
        
        trace_vmapped = self.nest_vmap(torch.trace, len(r_he.shape) - 2)
        c_map = (2.*dij*n_levels - trace_vmapped(r_ee @ torch.conj(r_ee.transpose(-2, -1))) + trace_vmapped(r_he @ torch.conj(r_he.transpose(-2, -1)))).real
        return torch.permute(c_map, (2,1,0))

    def nest_vmap(self, func, n_dims):
        for _ in range(n_dims):
            func = torch.vmap(func)
        return func

    def torch_h_set_mu(self,
        h_tensor: torch.Tensor,
        mu: torch.Tensor, # should be either a tensor of shape (..., 1) or (..., n_blocks)
        use_dot_split: bool = False
    ) -> torch.Tensor:
        assert h_tensor.shape[-1] >= 4, "At least one 4 x 4 block is required"
        n_blocks = h_tensor.shape[-1] // 4
        if mu.shape[-1] == 1:
            mu = mu.expand(*mu.shape[:-1], n_blocks)

        magnetic_field = torch.stack(
            [(h_tensor[..., i*4, i*4] + h_tensor[..., i*4 + 3, i*4 + 3]) / 2 for i in range(n_blocks)],
            dim=-1
        )

        if use_dot_split:
            dot_splits = torch.stack(
                [h_tensor[..., i*4, i*4] - h_tensor[..., (i+1)*4, (i+1)*4] for i in range(n_blocks - 1)],
                dim=-1
            )
            dot_splits = torch.cat([torch.zeros_like(dot_splits[..., :1]), dot_splits], dim=-1)
            mu = mu + dot_splits
        
        diag_idx = torch.arange(h_tensor.shape[-1], device=h_tensor.device)
        modified_h_tensor = h_tensor.clone()
        modified_h_tensor[..., diag_idx[0::4], diag_idx[0::4]] = -mu + magnetic_field
        modified_h_tensor[..., diag_idx[1::4], diag_idx[1::4]] = -mu - magnetic_field
        modified_h_tensor[..., diag_idx[2::4], diag_idx[2::4]] = mu - magnetic_field
        modified_h_tensor[..., diag_idx[3::4], diag_idx[3::4]] = mu + magnetic_field
        return modified_h_tensor

    def torch_h_set_b(self,
        h_tensor: torch.Tensor,
        b: torch.Tensor, # should be either a tensor of shape (..., 1) or (..., n_blocks)
        use_dot_split: bool = False
    ) -> torch.Tensor:
        assert h_tensor.shape[-1] >= 4, "At least one 4 x 4 block is required"
        n_blocks = h_tensor.shape[-1] // 4
        if b.shape[-1] == 1:
            b = b.expand(*b.shape[:-1], n_blocks)

        mu = torch.stack(
            [(h_tensor[..., i*4 + 2, i*4 + 2] + h_tensor[..., i*4 + 3, i*4 + 3]) / 2 for i in range(n_blocks)],
            dim=-1
        )

        if use_dot_split:
            dot_splits = torch.stack(
                [h_tensor[..., i*4, i*4] - h_tensor[..., (i+1)*4, (i+1)*4] for i in range(n_blocks - 1)],
                dim=-1
            )
            dot_splits = torch.cat([torch.zeros_like(dot_splits[..., :1]), dot_splits], dim=-1)
            mu = mu + dot_splits
        
        diag_idx = torch.arange(h_tensor.shape[-1], device=h_tensor.device)
        modified_h_tensor = h_tensor.clone()
        modified_h_tensor[..., diag_idx[0::4], diag_idx[0::4]] = -mu + b
        modified_h_tensor[..., diag_idx[1::4], diag_idx[1::4]] = -mu - b
        modified_h_tensor[..., diag_idx[2::4], diag_idx[2::4]] = mu - b
        modified_h_tensor[..., diag_idx[3::4], diag_idx[3::4]] = mu + b
        return modified_h_tensor

    def torch_w_matrix(self,
        matrix_dim: int,
        gamma: float = 0.1,
        n_levels: int = 1
    ) -> torch.Tensor:
        sg = np.sqrt(gamma)
        sg_tensor = torch.tensor([sg, sg, -sg, -sg])
        site = torch.diag(sg_tensor)
        if n_levels > 1:
            site = torch.kron(torch.eye(n_levels), site)

        W = torch.zeros((matrix_dim, matrix_dim))
        dim_0 = 4 * n_levels  # block_size * n_levels
        # left lead
        W[:dim_0, :dim_0] += site 
        # right lead
        W[-dim_0:, -dim_0:] += site 
        return W

    def torch_s_matrix(self,
        h_tensor: torch.Tensor,
        ef: torch.Tensor, # must be of shape matching h_tensor
        w_matrix: torch.Tensor, # of shape (matrix_dim, matrix_dim)
    ) -> torch.Tensor:
        
        w_matrix = w_matrix.to(h_tensor.device).to(h_tensor.dtype)
        S = torch.eye(h_tensor.shape[-1], device=h_tensor.device, dtype=h_tensor.dtype) * ef - h_tensor + w_matrix @ w_matrix.T * 0.5j
        Smat = torch.linalg.inv(S)
        Smat = w_matrix.T @ Smat @ w_matrix
        Smat = torch.eye(h_tensor.shape[-1], device=h_tensor.device, dtype=h_tensor.dtype) - Smat*1.j
        return Smat


class Encoder_simpleCNN(nn.Module): 
    def __init__(self,
                default_parameters,
                device,
                input_size: int,
                output_size: int):
        super().__init__()
        self.input_size = input_size
        self.pool = nn.MaxPool2d(3)
        self.conv1 = nn.Conv2d(16, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 32, 3)
        self.output_mlp = nn.Linear(32*64, output_size)
        self.output = nn.Sigmoid()
        mus = (default_parameters.mu_range[1]-default_parameters.mu_range[0])*au.Eh
        muf = default_parameters.mu_range[0]*au.Eh
        ts = (default_parameters.t_range[1]-default_parameters.t_range[0])*au.Eh
        tf = default_parameters.t_range[0]*au.Eh
        ls = (default_parameters.l_range[1]-default_parameters.l_range[0])
        lf = default_parameters.l_range[0]
        self.scale = (torch.tensor([mus,mus,mus,ts,ts,ls,ls])*1.2).to(device)
        self.offset = torch.tensor([muf,muf,muf,tf,tf,lf,lf]).to(device) - self.scale*0.1/1.2
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.output_mlp(x)
        x = self.output(x)*self.scale.expand_as(x) + self.offset.expand_as(x)
        return x


class Encoder_ViT(nn.Module):
    def __init__(self,
                 default_parameters,
                 device,
                 num_input_channels,
                 output_size: int):
        super().__init__()
        self.feature_extractor = create_model('vit_base_patch16_224', pretrained=False)
        self.feature_extractor.reset_classifier(0)  # Removes the final classification head
        # get and modify the existing patch embedding layer
        old_patch_embed = deepcopy(self.feature_extractor.patch_embed)
        self.feature_extractor.patch_embed.proj = nn.Conv2d(
            in_channels=num_input_channels,
            out_channels=old_patch_embed.proj.out_channels,
            kernel_size=old_patch_embed.proj.kernel_size,
            stride=old_patch_embed.proj.stride,
            padding=old_patch_embed.proj.padding,
            bias=old_patch_embed.proj.bias is not None
        )
        # copying RGB channels to new 16, no idea how this can be done better
        with torch.no_grad():
            self.feature_extractor.patch_embed.proj.weight[:,:3] = old_patch_embed.proj.weight
            self.feature_extractor.patch_embed.proj.weight[:,3:6] = old_patch_embed.proj.weight
            self.feature_extractor.patch_embed.proj.weight[:,6:9] = old_patch_embed.proj.weight
            self.feature_extractor.patch_embed.proj.weight[:,9:12] = old_patch_embed.proj.weight
            self.feature_extractor.patch_embed.proj.weight[:,12:15] = old_patch_embed.proj.weight
            self.feature_extractor.patch_embed.proj.weight[:,15] = old_patch_embed.proj.weight[:,0]
        #
        self.no_features = 768
        self.output_mlp = nn.Sequential(nn.Linear(self.no_features, 64), nn.ReLU(), 
                                        nn.Linear(64, 32), nn.ReLU(), 
                                        nn.Linear(32, output_size), nn.Sigmoid())
        mus = (default_parameters.mu_range[1]-default_parameters.mu_range[0])*au.Eh
        muf = default_parameters.mu_range[0]*au.Eh
        ts = (default_parameters.t_range[1]-default_parameters.t_range[0])*au.Eh
        tf = default_parameters.t_range[0]*au.Eh
        ls = (default_parameters.l_range[1]-default_parameters.l_range[0])
        lf = default_parameters.l_range[0]
        self.scale = (torch.tensor([mus,mus,mus,ts,ts,ls,ls])*1.2).to(device)
        self.offset = torch.tensor([muf,muf,muf,tf,tf,lf,lf]).to(device) - self.scale*0.1/1.2

    def parameters(self):
        #return self.output_mlp.parameters()
        return list(self.feature_extractor.parameters()) + list(self.output_mlp.parameters())

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.output_mlp(x)
        return x*self.scale.expand_as(x) + self.offset.expand_as(x)


class DeconvNet(nn.Module):

    def __init__(self,
                 input_dim : int,
                 num_input_channels : int,
                 num_output_channels : int,
                 num_hidden_channels : int,
                 act_fn : object = nn.GELU):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = input_dim**2
        c_hid = num_hidden_channels
        self.linear = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            act_fn()
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(num_input_channels, 2*c_hid, kernel_size=3, stride=2),
            act_fn(),
            nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, stride=2),
            act_fn(),
            nn.ConvTranspose2d(c_hid, c_hid, kernel_size=3, stride=2),
            act_fn(),
            nn.Conv2d(c_hid, num_output_channels, kernel_size=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], self.latent_dim)
        x = self.linear(x).reshape(x.shape[0], x.shape[1], self.input_dim, self.input_dim)
        x = self.net(x)*2.
        return x


class Decoder(nn.Module):
    
    def __init__(self, parameters, device, bypass=False):
        super().__init__()
        self.hamiltonian = Hamiltonian(parameters, device)
        self.transport = Transport(parameters, device)
        self.bypass = bypass
        if self.bypass:
            self.bypass_network = DeconvNet(12, 2, 16, 32)
            
    def parameters(self):
        return self.bypass_network.parameters()
    
    def forward(self, x):
        # x should be of shape [batch, no_parmeters]
        h_tensor = self.hamiltonian.generate(x)
        if self.bypass:
            return self.transport.torch_conductance_collection(h_tensor, x), self.bypass_network(torch.stack((h_tensor.real, h_tensor.imag), 1))
        else:
            return self.transport.torch_conductance_collection(h_tensor, x)


class Autoencoder(nn.Module):

    def __init__(self,
                 parameters,
                 device,
                 num_input_channels: int = 16,
                 input_size: [int,int] = [101,101],
                 latent_dim: int = 7,
                 bypass: bool = False):
        super().__init__()
        self.bypass = bypass
        #self.encoder = Encoder_simpleCNN(parameters.def_par, device, num_input_channels*input_size[0]*input_size[1], latent_dim).to(device) 
        self.encoder = Encoder_ViT(parameters.def_par, device, num_input_channels, latent_dim).to(device) 
        self.decoder = Decoder(parameters, device, self.bypass).to(device)
        
    def parameters(self):       
        if self.bypass:
            return list(self.encoder.parameters()) + list(self.decoder.parameters())
        else:
            return self.encoder.parameters()

    def train(self):
        self.encoder.train()
        self.decoder.train()

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def forward(self, x):
        z = self.encoder(x)
        if self.bypass:
            x_hat, x_prime = self.decoder(z)
            return [z, x_hat, x_prime]
        else:
            x_hat = self.decoder(z)
            return [z, x_hat]


class Experiments():

    def __init__(self, device, model, train_loader, test_loader, optimizer, criterion=nn.MSELoss(), bypass=False):
        self.device = device
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.bypass = bypass
        
    def loss(self, output, sample, parameters): 
        if self.bypass:
            return self.criterion(output[0], parameters), self.criterion(output[1], sample), self.criterion(output[2], sample)
        else:    
            return self.criterion(output[0], parameters), self.criterion(output[1], sample)
        
    def train(self, epoch_number):
        self.model.train()
        train_loss_p = 0.
        train_loss_c = 0.
        if self.bypass:
            train_loss_b = 0.       
        for batch_idx, (sample, sample0, parameters) in enumerate(self.train_loader):
            sample, sample0, parameters = sample.to(self.device), sample0.to(self.device), parameters.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(sample)
            if self.bypass:
                loss_p, loss_c, loss_b = self.loss(output, sample0, parameters)
            else:
                loss_p, loss_c = self.loss(output, sample0, parameters)
            # C reconstruction loss (+ bypass loss) only ??
            loss = loss_c + loss_b
            loss.backward()
            self.optimizer.step()
            train_loss_p += loss_p.item()
            train_loss_c += loss_c.item()
            if self.bypass:
                train_loss_b += loss_b.item()
        print('Train Epoch: {}'.format(epoch_number))
        train_loss_p /= len(self.train_loader)
        train_loss_c /= len(self.train_loader)
        if self.bypass:
            train_loss_b /= len(self.train_loader)
        print('\tTrain set: Average parameters prediction loss: {:.4f}'.format(train_loss_p))
        print('\tTrain set: Average C reconstruction loss: {:.4f}'.format(train_loss_c))
        if self.bypass:
            print('\tTrain set: Average bypass reconstruction loss: {:.4f}'.format(train_loss_b))
        return train_loss_p, train_loss_c
    
    def test(self):
        self.model.eval()
        test_loss_p = 0.
        test_loss_c = 0.
        for batch_idx, (sample, sample0, parameters) in enumerate(self.test_loader):
            sample, sample0, parameters = sample.to(self.device), sample0.to(self.device), parameters.to(self.device)
            output = self.model(sample)
            if self.bypass:
                loss_p, loss_c, _ = self.loss(output, sample0, parameters)
            else:
                loss_p, loss_c = self.loss(output, sample0, parameters)
            test_loss_p += loss_p.item()
            test_loss_c += loss_c.item()
        test_loss_p /= len(self.test_loader)
        test_loss_c /= len(self.test_loader)
        print('\tTest set: Average parameters prediction loss: {:.4f}'.format(test_loss_p))
        print('\tTest set: Average C reconstruction loss: {:.4f}'.format(test_loss_c))
        return test_loss_p, test_loss_c

    def run_training(self, no_epochs):
        train_loss = []
        validation_loss = []
        for epoch_number in range(1, no_epochs+1):
            train_loss.append([*self.train(epoch_number)])
            validation_loss.append([*self.test()])
        return train_loss, validation_loss
    
    def get_prediction(self, i):
        self.model.eval()
        sample, sample0, parameters = next(iter(self.test_loader))
        sample, parameters = sample.to(self.device), parameters.to(self.device)
        output = self.model(sample)
        return output[1][i].detach().cpu(), sample0[i]
    

def plot_loss(train_loss, validation_loss, title):
    plt.grid(True)
    plt.xlabel("subsequent epochs")
    plt.ylabel('average loss')
    plt.plot(range(1, len(train_loss)+1), train_loss[:,0], 'o-', label='train p pred. loss')
    plt.plot(range(1, len(train_loss)+1), train_loss[:,1], 'o-', label='train C rec. loss')
    plt.plot(range(1, len(validation_loss)+1), validation_loss[:,0], 'o-', label='val p pred. loss')
    plt.plot(range(1, len(validation_loss)+1), validation_loss[:,1], 'o-', label='val C rec. loss')
    plt.legend()
    plt.title(title)
    plt.savefig(os.path.join('./', 'loss.png'), bbox_inches='tight', dpi=200)
    plt.close()


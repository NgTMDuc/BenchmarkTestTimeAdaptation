from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import torchvision
import math
import numpy as np 
import matplotlib.pyplot as plt
import torch.nn.functional as F
from einops import rearrange

class NU(nn.Module):
    def __init__(self, model, args, optimizer, 
                 steps = 1, episodic = False, deyo_margin = 0.5 * math.log(1000), 
                 margin_e0 = 0.4 * math.log(1000)):
        super().__init__()
        self.model = model
        self.args = args
        self.optimizer = optimizer

        self.steps = steps
        self.episodic = episodic

        self.deyo_margin = deyo_margin
        self.margin_e0 = margin_e0

        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(self.model, self.optimizer)

        self.device =  next(self.model.parameters()).device

        self.embed, self.encoder1, self.encoder2, self.post_layer ,self.fc = self.split_model()
    
    def split_model(self):
        fc = list(self.model.children())[-1]

        if self.args.MODEL.ARCH == "Hendrycks2020AugMix_ResNeXt":
            embed = nn.Sequential(*list(self.model.children())[0:2])
            blocks = nn.Sequential(*list(self.model.children())[2:5])
            post_layer = nn.Sequential(list(self.model.children())[5])

        elif self.args.MODEL.ARCH == "WideResNet":
            embed = nn.Sequential(list(self.model.children())[0])
            blocks = nn.Sequential(*list(self.model.children())[1:4])
            post_layer = nn.Sequential(*list(self.model.children())[4:5])

        elif self.args.MODEL.ARCH == "vit":
            embed = nn.Sequential(*list(self.model.children())[0:4])
            blocks = nn.Sequential(*list(list(self.model.children())[4].children()))
            post_layer = nn.Sequential(*list(self.model.children())[5:-1])

        elif self.args.MODEL.ARCH == "officehome_shot" or self.args.MODEL.ARCH == "domainnet126_shot":
            embed = nn.Sequential(*list(self.model.netF.children())[0:4])
            blocks = nn.Sequential(*list(self.model.netF.children())[4:8])
            post_layer = nn.Sequential((*list(self.model.netF.children())[8:], self.model.netB))

        n_blocks = len(blocks)
        assert n_blocks >= self.args.PROPOSAL.LAYER, f"There are only {n_blocks} blocks in model"
        encoder1 = blocks[:self.args.PROPOSAL.LAYER]
        
        if self.args.PROPOSAL.LAYER == n_blocks:
            encoder2 = nn.Identity()
        else:
            encoder2 = blocks[self.args.PROPOSAL.LAYER :]

        return embed, encoder1, encoder2, post_layer, fc
    
    def get_filter(self, x):
        with torch.no_grad():
            embedding = self.embed(x)
            feature_ori = self.encoder1(embedding)
        
        mean, std = self.get_style(feature_ori)
        
        sqrtvar_mu = self.sqrtvar(mean)
        sqrtvar_std = self.sqrtvar(std)

        beta = self._reparameterize(mean, sqrtvar_mu)
        gamma = self._reparameterize(std, sqrtvar_std)
        num_dims = feature_ori.dim()
        shape = (feature_ori.shape[0], feature_ori.shape[1]) + (1,) * (num_dims - 2)
        norms = (feature_ori - mean.reshape(shape)) / std.reshape(shape)
        x_trans = norms * gamma.reshape(shape) + beta.reshape(shape)
        feature_ori = self.encoder2(feature_ori)
        x_trans = self.encoder2(x_trans)

        if self.args.MODEL.ARCH == "vit":
            feature_ori = self.model.pool(feature_ori)
            x_trans = self.model.pool(x_trans)

        feature_ori = self.post_layer(feature_ori)
        x_trans = self.post_layer(x_trans) 
        
        if self.args.MODEL.ARCH == "WideResNet":
            feature_ori = F.avg_pool2d(feature_ori, 8)
            x_trans = F.avg_pool2d(x_trans, 8)
        
        
        with torch.no_grad():
            output_ori = self.fc(feature_ori.view(feature_ori.size(0), -1)).softmax(1)
            output_aug = self.fc(x_trans.view(feature_ori.size(0), -1)).softmax(1)
        
        predictions = torch.argmax(output_ori, dim=1)
        plpd_trans = (torch.gather(output_ori, dim = 1, index = predictions.reshape(-1, 1)) - torch.gather(output_aug, dim = 1, index = predictions.reshape(-1, 1))) 
        plpd_trans = plpd_trans.reshape(-1).cpu()
        all_below_threshold4 = torch.where(plpd_trans < self.args.PROPOSAL.NEW_MARGIN)

        return all_below_threshold4, plpd_trans
    def forward(self, x):
        if self.episodic:
            self.reset()
        
        filter_ids0, plpd_new = self.get_filter(x)
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, filter_ids0, plpd_new)
            
        
        return outputs

    @torch.enable_grad()
    def forward_and_adapt(self, x, filter_ids0, plpd_new):
        outputs = self.model(x)
        self.optimizer.zero_grad()
        entropys = softmax_entropy(outputs)

        if self.args.DEYO.FILTER_ENT:
            filter_ids1 = torch.where((entropys < self.deyo_margin))[0]
        else:
            filter_ids1 = torch.where((entropys <= math.log(1000)))[0]
        
        x_prime = x.detach()

        if self.args.DEYO.AUG_TYPE == "occ":
            first_mean = x_prime.view(x_prime.shape[0], x_prime.shape[1], -1).mean(dim=2)
            final_mean = first_mean.unsqueeze(-1).unsqueeze(-1)
            occlusion_window = final_mean.expand(-1, -1, self.args.DEYO.OCCLUSION_SIZE, self.args.DEYO.OCCLUSION_SIZE)
            x_prime[:, :, self.args.DEYO.ROW_START:self.args.DEYO.ROW_START+self.args.DEYO.OCCLUSION_SIZE,self.args.DEYO.COLUMN_START:self.args.DEYO.COLUMN_START+self.args.DEYO.OCCLUSION_SIZE] = occlusion_window
        
        elif self.args.DEYO.AUG_TYPE == "patch":
            resize_t = torchvision.transforms.Resize(((x.shape[-1]//self.args.DEYO.PATCH_LEN)*self.args.DEYO.PATCH_LEN,(x.shape[-1]//self.args.DEYO.PATCH_LEN)*self.args.DEYO.PATCH_LEN))
            resize_o = torchvision.transforms.Resize((x.shape[-1],x.shape[-1]))
            x_prime = resize_t(x_prime)
            x_prime = rearrange(x_prime, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w', ps1=self.args.DEYO.PATCH_LEN, ps2=self.args.DEYO.PATCH_LEN)
            perm_idx = torch.argsort(torch.rand(x_prime.shape[0],x_prime.shape[1]), dim=-1)
            x_prime = x_prime[torch.arange(x_prime.shape[0]).unsqueeze(-1),perm_idx]
            x_prime = rearrange(x_prime, 'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)', ps1=self.args.DEYO.PATCH_LEN, ps2=self.args.DEYO.PATCH_LEN)
            x_prime = resize_o(x_prime)
        elif self.args.DEYO.AUG_TYPE == "pixel":
            x_prime = rearrange(x_prime, 'b c h w -> b c (h w)')
            x_prime = x_prime[:,:,torch.randperm(x_prime.shape[-1])]
            x_prime = rearrange(x_prime, 'b c (ps1 ps2) -> b c ps1 ps2', ps1=x.shape[-1], ps2=x.shape[-1])
        
        with torch.no_grad():
            outputs_prime = self.model(x_prime)
        
        prob_outputs = outputs.softmax(1)
        prob_outputs_prime = outputs_prime.softmax(1)

        cls1 = prob_outputs.argmax(dim = 1)

        plpd = torch.gather(prob_outputs, dim=1, index=cls1.reshape(-1,1)) - torch.gather(prob_outputs_prime, dim=1, index=cls1.reshape(-1,1))
        plpd = plpd.reshape(-1)
        if self.args.DEYO.FILTER_PLPD:
            filter_ids2 = torch.where(plpd > self.args.DEYO.PLPD_THRESHOLD)[0]
        else:
            filter_ids2 = torch.where(plpd >= -2.0)[0]
        
        combined = torch.cat([filter_ids0[0].to(self.device), filter_ids1, filter_ids2])
        idx = torch.unique(combined)

        if self.args.DEYO.REWEIGHT_ENT or self.args.DEYO.REWEIGHT_PLPD:
            coeff = (
                self.args.DEYO.REWEIGHT_ENT * (1 / (torch.exp(((entropys.clone().detach()) - self.margin_e0)))) + 
                self.args.DEYO.REWEIGHT_PLPD * (1 / (torch.exp(-1. * plpd.clone().detach()))) + 
                1 / (torch.exp(plpd_new.to(self.device).clone().detach() - self.args.PROPOSAL.NEW_MARGIN_E0))
            )

            entropys = entropys.mul(coeff)
        
        loss = entropys[idx].mean(0)

        if self.args.PROPOSAL.USE_BAD:
            filter_bad = torch.where(plpd < self.args.PROPOSAL.BAD_MARGIN)
            entropys_backward = entropys.clone().detach()[
                torch.unique(torch.cat([filter_bad[0], filter_ids1]))
            ]
            plpd_backward = plpd.clone().detach()[
                torch.unique(torch.cat([filter_bad[0], filter_ids1]))
            ]

            bad_coeff = self.args.PROPOSAL.ALPHA * (1 / (torch.exp(((entropys_backward.clone().detach()) - self.margin_e0))))
            entropys_backward = entropys_backward.mul(bad_coeff).mul(-1)

            loss = loss + entropys_backward.mean(0)
        
        loss.backward()
        self.optimizer.step()
        del x_prime

        return outputs


    def get_style(self, x):
        if len(x.shape) == 4:
            mu = x.mean(dim = [2, 3], keepdim = False)
            var = x.var(dim = [2, 3], keepdim = False)
            var = (var + 1e-6).sqrt()
        elif len(x.shape) == 3:
            mu = x.mean(dim = 2, keepdim = False)
            var = x.var(dim = 2, keepdim = False)
            var = (var + 1e-6).sqrt()
        return mu, var
    
    def sqrtvar(self, x):
        t = (x.var(dim = 0, keepdim = True) + 1e-6).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t
    
    def  _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * 0.1
        return mu + epsilon * std 

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.ema = None

    @staticmethod
    def configure_model(model):
        """Configure model for use with DeYO."""
        # train mode, because DeYO optimizes the model to minimize entropy
        model.train()
        # disable grad, to (re-)enable only what DeYO updates
        model.requires_grad_(False)
        # configure norm for DeYO updates: enable grad + force batch statisics (this only for BN models)
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
            if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)
        return model
    
    @staticmethod
    def collect_params(model):
        """Collect the affine scale + shift parameters from norm layers.
        Walk the model's modules and collect all normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        for nm, m in model.named_modules():
            # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
            if 'layer4' in nm:
                continue
            if 'blocks.9' in nm:
                continue
            if 'blocks.10' in nm:
                continue
            if 'blocks.11' in nm:
                continue
            if 'norm.' in nm:
                continue
            if nm in ['norm']:
                continue

            if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")

        return params, names
    @staticmethod
    def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
        """Restore the model and optimizer states from copies."""
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(optimizer_state)

    @staticmethod
    def copy_model_and_optimizer(model, optimizer):
        """Copy the model and optimizer states for resetting after adaptation."""
        model_state = deepcopy(model.state_dict())
        optimizer_state = deepcopy(optimizer.state_dict())
        return model_state, optimizer_state
@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    #temprature = 1.1 #0.9 #1.2
    #x = x ** temprature #torch.unsqueeze(temprature, dim=-1)
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)
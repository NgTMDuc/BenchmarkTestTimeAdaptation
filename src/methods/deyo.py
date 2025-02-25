from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import torchvision
import math
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange

class DeYO(nn.Module):
    def __init__(self, model, args, optimizer, steps = 1, episodic = False, deyo_margin=0.5*math.log(1000), margin_e0=0.4*math.log(1000)):
        super().__init__()
        self.model = model 
        self.optimizer = optimizer
        self.args = args  
        
        self.steps = steps
        self.episodic = episodic
        
        self.deyo_margin = deyo_margin
        self.margin_e0 = margin_e0
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()
        
        for _ in range(self.steps):
            outputs = forward_and_adapt_deyo(x, self.model, self.args, self.optimizer, self.deyo_margin, self.margin_e0)
        
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
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

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    #temprature = 1.1 #0.9 #1.2
    #x = x ** temprature #torch.unsqueeze(temprature, dim=-1)
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt_deyo(x, model, args, optimizer, deyo_margin, margin):
    outputs = model(x)
    optimizer.zero_grad()
    entropys = softmax_entropy(outputs)
    if args.DEYO.FILTER_ENT:
        filter_ids_1 = torch.where((entropys < deyo_margin))[0]
    else:    
        filter_ids_1 = torch.where((entropys <= math.log(1000)))[0]
    # entropys = entropys[filter_ids_1]
    # if len(entropys) == 0:
        # return outputs
    
    # x_prime = x[filter_ids_1]
    x_prime = x.detach()
    if args.DEYO.AUG_TYPE=='occ':
        first_mean = x_prime.view(x_prime.shape[0], x_prime.shape[1], -1).mean(dim=2)
        final_mean = first_mean.unsqueeze(-1).unsqueeze(-1)
        occlusion_window = final_mean.expand(-1, -1, args.DEYO.OCCLUSION_SIZE, args.DEYO.OCCLUSION_SIZE)
        x_prime[:, :, args.DEYO.ROW_START:args.DEYO.ROW_START+args.DEYO.OCCLUSION_SIZE,args.DEYO.COLUMN_START:args.DEYO.COLUMN_START+args.DEYO.OCCLUSION_SIZE] = occlusion_window
    elif args.DEYO.AUG_TYPE=='patch':
        resize_t = torchvision.transforms.Resize(((x.shape[-1]//args.DEYO.PATCH_LEN)*args.DEYO.PATCH_LEN,(x.shape[-1]//args.DEYO.PATCH_LEN)*args.DEYO.PATCH_LEN))
        resize_o = torchvision.transforms.Resize((x.shape[-1],x.shape[-1]))
        x_prime = resize_t(x_prime)
        x_prime = rearrange(x_prime, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w', ps1=args.DEYO.PATCH_LEN, ps2=args.DEYO.PATCH_LEN)
        perm_idx = torch.argsort(torch.rand(x_prime.shape[0],x_prime.shape[1]), dim=-1)
        x_prime = x_prime[torch.arange(x_prime.shape[0]).unsqueeze(-1),perm_idx]
        x_prime = rearrange(x_prime, 'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)', ps1=args.DEYO.PATCH_LEN, ps2=args.DEYO.PATCH_LEN)
        x_prime = resize_o(x_prime)
    elif args.DEYO.AUG_TYPE=='pixel':
        x_prime = rearrange(x_prime, 'b c h w -> b c (h w)')
        x_prime = x_prime[:,:,torch.randperm(x_prime.shape[-1])]
        x_prime = rearrange(x_prime, 'b c (ps1 ps2) -> b c ps1 ps2', ps1=x.shape[-1], ps2=x.shape[-1])
    with torch.no_grad():
        outputs_prime = model(x_prime)

    prob_outputs = outputs.softmax(1)
    prob_outputs_prime = outputs_prime.softmax(1)

    cls1 = prob_outputs.argmax(dim=1)

    plpd = torch.gather(prob_outputs, dim=1, index=cls1.reshape(-1,1)) - torch.gather(prob_outputs_prime, dim=1, index=cls1.reshape(-1,1))
    plpd = plpd.reshape(-1)

    if args.DEYO.FILTER_PLPD:
        filter_ids_2 = torch.where(plpd > args.DEYO.PLPD_THRESHOLD)[0]
    else:
        filter_ids_2 = torch.where(plpd >= -2.0)[0]
    # entropys = entropys[filter_ids_2]
    combined_filter_ids = torch.tensor(list(set(filter_ids_1.tolist()) & set(filter_ids_2.tolist())))
    plpd_return = plpd.clone().detach()
    entropys_return = entropys.clone().detach()
    # print(len(combined_filter_ids))
    if len(combined_filter_ids) == 0:
        return outputs, entropys_return, plpd_return
    entropys = entropys[combined_filter_ids]
    # if len(entropys) == 0:
    #     del x_prime
    #     del plpd
    #     return outputs
    # plpd = plpd[filter_ids_2]
    # plpd_return = plpd.clone().detach()
    plpd = plpd[combined_filter_ids]

    if args.DEYO.REWEIGHT_ENT or args.DEYO.REWEIGHT_PLPD:
        coeff = (args.DEYO.REWEIGHT_ENT * (1 / (torch.exp(((entropys.clone().detach()) - margin)))) +
                 args.DEYO.REWEIGHT_PLPD * (1 / (torch.exp(-1. * plpd.clone().detach())))
                )            
        entropys = entropys.mul(coeff)
    loss = entropys.mean(0)

    if len(entropys) != 0:
        loss.backward()
        optimizer.step()
    optimizer.zero_grad()

    del x_prime
    # del plpd
    return outputs, entropys_return, plpd_return



def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state
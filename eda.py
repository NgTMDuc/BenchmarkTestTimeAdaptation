import argparse
from copy import deepcopy
import copy
import torch
import torch.nn as nn
import torch.jit
import torchvision
import math
import numpy as np
import logging
import matplotlib.pyplot as plt
from einops import rearrange
from datetime import datetime
import numpy as np
import json

from src.methods import *
from src.models.load_model import load_model
from src.utils import get_accuracy, get_args, evaluate_model
from src.utils.conf import cfg, load_cfg_fom_args, get_num_classes, get_domain_sequence
import numpy as np
import torch
from iopath.common.file_io import g_pathmgr
from yacs.config import CfgNode as CfgNode
import os 
from tqdm import tqdm 

# Global config object (example usage: from core.config import cfg)
_C = CfgNode()
cfg = _C

# ---------------------------------- Misc options --------------------------- #

# Setting - see README.md for more information
_C.bad = False
# Data directory_
_C.DATA_DIR = "../DATA/"

# Weight directory
_C.CKPT_DIR = "./ckpt/"

# Output directory
_C.OUTPUT = "./output"

# Path to a specific checkpoint
_C.CKPT_PATH = ""

# Log destination (in SAVE_DIR)
_C.LOG_DEST = "log.txt"

# Log datetime
_C.LOG_TIME = ''

# Seed to use. If None, seed is not set!
# Note that non-determinism is still present due to non-deterministic GPU ops.
_C.RNG_SEED = 2024

# Deterministic experiments.
_C.DETERMINISM = False

# The num_workers argument to use in the data loaders
_C.WORKERS = 4

# ----------------------------- Model options ------------------------------- #
_C.MODEL = CfgNode()

# Check https://github.com/RobustBench/robustbench or https://pytorch.org/vision/0.14/models.html for available models
_C.MODEL.ARCH = 'resnet50-bn'

# Type of pre-trained weights for torchvision models. See: https://pytorch.org/vision/0.14/models.html
_C.MODEL.WEIGHTS = "IMAGENET1K_V1"

# Inspect the cfgs directory to see all possibilities
_C.MODEL.ADAPTATION = 'deyo'

# Reset the model before every new batch
_C.MODEL.EPISODIC = False

_C.MODEL.CONTINUAL = 'Fully'

# ----------------------------- Corruption options -------------------------- #
_C.CORRUPTION = CfgNode()

# Dataset for evaluation
_C.CORRUPTION.DATASET = 'waterbirds'

_C.CORRUPTION.SOURCE_DATASET = 'waterbirds'

_C.CORRUPTION.SOURCE_DOMAIN = 'origin'
_C.CORRUPTION.SOURCE_DOMAINS = ['origin']
# Check https://github.com/hendrycks/robustness for corruption details
_C.CORRUPTION.TYPE = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                      'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                      'snow', 'frost', 'fog', 'brightness', 'contrast',
                      'elastic_transform', 'pixelate', 'jpeg_compression']
_C.CORRUPTION.SEVERITY = [0]

# Number of examples to evaluate (10000 for all samples in CIFAR-C)
# For ImageNet-C, RobustBench loads a list containing 5000 samples.
# If num_ex is larger than 5000 all images (50,000) are loaded and then subsampled to num_ex
_C.CORRUPTION.NUM_EX = -1

# ------------------------------- Batch norm options ------------------------ #
_C.BN = CfgNode()

# BN alpha (1-alpha) * src_stats + alpha * test_stats
_C.BN.ALPHA = 0.1

# ------------------------------- Optimizer options ------------------------- #
_C.OPTIM = CfgNode()

# Number of updates per batch
_C.OPTIM.STEPS = 1

# Learning rate
_C.OPTIM.LR = 1e-3

# Choices: Adam, SGD
_C.OPTIM.METHOD = 'SGD'

# Beta
_C.OPTIM.BETA = 0.9

# Momentum
_C.OPTIM.MOMENTUM = 0.9

# Momentum dampening
_C.OPTIM.DAMPENING = 0.0

# Nesterov momentum
_C.OPTIM.NESTEROV = True

# L2 regularization
_C.OPTIM.WD = 0.0

# ------------------------------------- T3A options ------------------------- #
_C.T3A = CfgNode()
_C.T3A.FILTER_K = 10

# --------------------------------- Mean teacher options -------------------- #
_C.M_TEACHER = CfgNode()

# Mean teacher momentum for EMA update
_C.M_TEACHER.MOMENTUM = 0.999

# --------------------------------- Contrastive options --------------------- #
_C.CONTRAST = CfgNode()

# Temperature term for contrastive learning
_C.CONTRAST.TEMPERATURE = 0.1

# Output dimension of projector
_C.CONTRAST.PROJECTION_DIM = 128

# Contrastive mode
_C.CONTRAST.MODE = 'all'

# --------------------------------- CoTTA options --------------------------- #
_C.COTTA = CfgNode()

# Restore probability
_C.COTTA.RST = 0.01

# Average probability for TTA
_C.COTTA.AP = 0.92

# --------------------------------- GTTA options ---------------------------- #
_C.GTTA = CfgNode()

_C.GTTA.STEPS_ADAIN = 1
_C.GTTA.PRETRAIN_STEPS_ADAIN = 20000
_C.GTTA.LAMBDA_MIXUP = 1 / 3
_C.GTTA.USE_STYLE_TRANSFER = False

# --------------------------------- RMT options ----------------------------- #
_C.RMT = CfgNode()

_C.RMT.LAMBDA_CE_SRC = 1.0
_C.RMT.LAMBDA_CE_TRG = 1.0
_C.RMT.LAMBDA_CONT = 1.0
_C.RMT.NUM_SAMPLES_WARM_UP = 50000

# --------------------------------- AdaContrast options --------------------- #
_C.ADACONTRAST = CfgNode()

_C.ADACONTRAST.QUEUE_SIZE = 16384
_C.ADACONTRAST.CONTRAST_TYPE = "class_aware"
_C.ADACONTRAST.CE_TYPE = "standard"  # ["standard", "symmetric", "smoothed", "soft"]
_C.ADACONTRAST.ALPHA = 1.0  # lambda for classification loss
_C.ADACONTRAST.BETA = 1.0  # lambda for instance loss
_C.ADACONTRAST.ETA = 1.0  # lambda for diversity loss

_C.ADACONTRAST.DIST_TYPE = "cosine"  # ["cosine", "euclidean"]
_C.ADACONTRAST.CE_SUP_TYPE = "weak_strong"  # ["weak_all", "weak_weak", "weak_strong", "self_all"]
_C.ADACONTRAST.REFINE_METHOD = "nearest_neighbors"
_C.ADACONTRAST.NUM_NEIGHBORS = 10

# --------------------------------- LAME options ----------------------------- #
_C.LAME = CfgNode()

_C.LAME.AFFINITY = "rbf"
_C.LAME.KNN = 5
_C.LAME.SIGMA = 1.0
_C.LAME.FORCE_SYMMETRY = False

# --------------------------------- SAR options ----------------------------- #
_C.SAR = CfgNode()
_C.SAR.RESET_CONSTANT = 0.005
_C.SAR.E_MARGIN_COE = 0.4

# --------------------------------- EATA options ---------------------------- #
_C.EATA = CfgNode()

# Fisher alpha
_C.EATA.FISHER_ALPHA = 2000

# Number of samples for ewc regularization
_C.EATA.NUM_SAMPLES = 2000

# Diversity margin
_C.EATA.D_MARGIN = 0.05

_C.EATA.E_MARGIN_COE = 0.4
# ------------------------------- DEYO options ---------------------------- #
_C.DEYO = CfgNode()
_C.DEYO.MARGIN = 0.5
_C.DEYO.MARGIN_E0 = 0.4
_C.DEYO.FILTER_ENT = 1
_C.DEYO.FILTER_PLDPD = 1
_C.DEYO.REWEIGHT_ENT = 1
_C.DEYO.REWEIGHT_PLPD = 1
_C.DEYO.PLPD_THRESHOLD = 0.2
_C.DEYO.AUG_TYPE = "patch"
_C.DEYO.OCCLUSION_SIZE = 112
_C.DEYO.ROW_START = 56
_C.DEYO.COLUMN_START = 56
_C.DEYO.PATCH_LEN = 4
# ------------------------------- Source options ---------------------------- #
_C.SOURCE = CfgNode()

# Number of workers for source data loading
_C.SOURCE.NUM_WORKERS = 4

# Percentage of source samples used
_C.SOURCE.PERCENTAGE = 1.0  # [0, 1] possibility to reduce the number of source samples

# ------------------------------- Testing options --------------------------- #
_C.TEST = CfgNode()

# Number of workers for test data loading
_C.TEST.NUM_WORKERS = 4

# Batch size for evaluation (and updates for norm + tent)
_C.TEST.BATCH_SIZE = 64

# If the batch size is 1, a sliding window approach can be applied by setting window length > 1
_C.TEST.WINDOW_LENGTH = 1

_C.TEST.EPOCH = 1

# Number of augmentations for methods relying on TTA (test time augmentation)
_C.TEST.N_AUGMENTATIONS = 32

# ------------------------------- NRC options -------------------------- #
_C.NRC = CfgNode()
_C.NRC.K = 5
_C.NRC.KK = 5
_C.NRC.EPSILION = 1e-5

# ------------------------------- SHOT options -------------------------- #
_C.SHOT = CfgNode()
_C.SHOT.EPSILION = 1e-5
_C.SHOT.CLS_PAR = 0.3
_C.SHOT.DISTANCE = 'cosine'
_C.SHOT.THRESHOLD = 0
_C.SHOT.ENT_PAR = 1

# ------------------------------- PLUE options -------------------------- #
_C.PLUE = CfgNode()
_C.PLUE.CTR= True
_C.PLUE.NUM_NEIGHBORS = 10
_C.PLUE.TEMPORAL_LENGTH = 5
_C.PLUE.TEMPERATURE = 0.07
_C.PLUE.LABEL_REFINEMENT = True
_C.PLUE.NEG_L = True
_C.PLUE.REWEIGHTING = True

# --------------------------------- CUDNN options --------------------------- #
_C.CUDNN = CfgNode()

# Benchmark to select fastest CUDNN algorithms (best for fixed input sizes)
_C.CUDNN.BENCHMARK = True

# --------------------------------- Default config -------------------------- #
_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()

class EDA(nn.Module):
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
        
        self.device = next(self.model.module.parameters()).device
        # self.decode = self.model.children()[0:4]
        self.decoder1 = self.model.children()[0:6]
        self.decoder2 = self.model.children()[6:9]
        self.fc = self.model.children()[9]

    def get_style(self, x):
        mu = x.mean(dim = [2, 3], keepdim = False)
        var = x.var(dim = [2, 3], keepdim = False)
        var = (var + 1e-6).sqrt()
        # mu, var = mu.detach(), var.detach()
        return mu, var
    
    def sqrtvar(self, x):
        t = (x.var(dim = 0, keepdim = True) + 1e-6).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t
    
    def  _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * 0.1
        return mu + epsilon * std 
    
    def get_filter(self, x):
        with torch.no_grad():
            feature_ori = self.decoder1(x)
        
        mean, std = self.get_style(feature_ori)

        sqrtvar_mu = self.sqrtvar(mean)
        sqrtvar_std = self.sqrtvar(std)

        beta = self._reparameterize(mean, sqrtvar_mu)
        gamma = self._reparameterize(std, sqrtvar_std) 

        norms = (feature_ori - mean.reshape(feature_ori.shape[0],feature_ori.shape[1],1,1))/std.reshape(feature_ori.shape[0],feature_ori.shape[1],1,1)
        x_trans = norms * gamma.reshape(feature_ori.shape[0],feature_ori.shape[1],1,1) + beta.reshape(feature_ori.shape[0],feature_ori.shape[1],1,1)
        with torch.no_grad():
            output_ori = self.fc(self.decoder2(feature_ori)).softmax(1)
            output_aug = self.fc(self.decoder2(x_trans)).softmax(1)
        
        predictions = torch.argmax(output_ori, dim=1)
        plpd_trans = (torch.gather(output_ori, dim = 1, index = predictions.reshape(-1, 1)) - torch.gather(output_aug, dim = 1, index = predictions.reshape(-1, 1))) 
        plpd_trans = plpd_trans.reshape(-1).cpu()
        all_below_threshold4 = torch.where(plpd_trans < self.args.DEYO.PLPD_THRESHOLD/2)

        return all_below_threshold4

    def forward(self, x):
        if self.episodic:
            self.reset()
        if self.args.propose:
            filter0, plpd_trans = self.get_filter(x)
        else:
            filter0, plpd_trans = None, None
        for _ in range(self.steps):
            # filter0 = self.get_filter(x)
            outputs, return_entropys, filter_ids_1, filter_ids_2, return_entropys, plpd = forward_and_adapt_deyo(x, 
                                                                                                                 self.model, self.args, self.optimizer, 
                                                                                                                 self.deyo_margin, self.margin_e0, filter0, 
                                                                                                                 plpd_trans)
        
        return outputs, return_entropys, filter_ids_1, filter_ids_2, return_entropys, plpd

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
def forward_and_adapt_deyo(x, model, args, optimizer, deyo_margin, margin, filter = None, plpd_trans = None):
    outputs = model(x)
    optimizer.zero_grad()
    entropys = softmax_entropy(outputs)

    return_entropys = copy.deepcopy(entropys)
    if args.DEYO.FILTER_ENT:
        filter_ids_1 = torch.where((entropys < deyo_margin))
    else:    
        filter_ids_1 = torch.where((entropys <= math.log(1000)))
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

    if args.DEYO.FILTER_PLDPD:
        filter_ids_2 = torch.where(plpd > args.DEYO.PLPD_THRESHOLD)
    else:
        filter_ids_2 = torch.where(plpd >= -2.0)

    if filter is None:
        combined = torch.cat([filter_ids_1, filter_ids_2])
    else:
        combined = torch.cat([filter, filter_ids_1, filter_ids_2])
    
    idx = torch.unique(combined)
    if args.DEYO.REWEIGHT_ENT or args.DEYO.REWEIGHT_PLPD:
        coeff = (args.DEYO.REWEIGHT_ENT * (1 / (torch.exp(((entropys.clone().detach()) - margin)))) +
                 args.DEYO.REWEIGHT_PLPD * (1 / (torch.exp(-1. * plpd.clone().detach()))) + 
                 1 * 1 / (torch.exp(plpd_trans.clone().detach() - 0.8))
                )            
        entropys = entropys.mul(coeff)
    loss = entropys[idx].mean(0)

    if args.bad:
        filter_bad = torch.where(plpd <= 0.5 * args.DEYO.PLPD_THRESHOLD)
        entropys_backward = copy.deepcopy(entropys)[
            torch.unique(torch.cat[filter_bad, filter_ids_1])
            ]
        plpd_backward = copy.deepcopy(plpd)[
            torch.unique(torch.cat[filter_bad, filter_ids_1])
            ]

        coeff1 = 0.5 * (1 / (torch.exp(((entropys_backward.clone().detach()) - margin))))

        entropys_backward = entropys_backward.mul(coeff1)
        entropys_backward = entropys_backward.mul(-1)

        loss = loss + entropys_backward.mean(0)

    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    del x_prime
    del plpd

    return outputs, return_entropys, filter_ids_1, filter_ids_2, plpd



def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state
def setup_optimizer(params, cfg):
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                          lr=cfg.OPTIM.LR,
                          betas=(cfg.OPTIM.BETA, 0.999),
                          weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                         lr=cfg.OPTIM.LR,
                         momentum=cfg.OPTIM.MOMENTUM,
                         dampening=cfg.OPTIM.DAMPENING,
                         weight_decay=cfg.OPTIM.WD,
                         nesterov=cfg.OPTIM.NESTEROV)
    else:
        raise NotImplementedError
if __name__ == "__main__":
    num_classes = get_num_classes(dataset_name = cfg.CORRUPTION.DATASET)
    base_model = load_model(model_name=cfg.MODEL.ARCH, 
                            checkpoint_dir=os.path.join(cfg.CKPT_DIR, 'models'),
                            domain=cfg.CORRUPTION.SOURCE_DOMAIN, 
                            cfg = cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model.to(device)

    model = EDA.configure_model(base_model)
    params, param_names = EDA.collect_params(model)
    optimizer = setup_optimizer(params, cfg)
    # optimizer = None 
    model = DeYO(model, cfg, optimizer, steps = cfg.OPTIM.STEPS, episodic=cfg.MODEL.EPISODIC,
                      deyo_margin = math.log(num_classes) * cfg.DEYO.MARGIN,
                      margin_e0 = math.log(num_classes) * cfg.DEYO.MARGIN_E0
                      )
    

    if cfg.CORRUPTION.DATASET in {"domainnet126", "officehome"}:
        dom_names_all = get_domain_sequence(cfg.CORRUPTION.DATASET, cfg.CORRUPTION.SOURCE_DOMAIN)
    else:
        dom_names_all = cfg.CORRUPTION.TYPE
    logger = logging.getLogger(__name__)
    logger.info(f"Using the following domain sequence: {dom_names_all}")

    dom_names_loop = dom_names_all
    severities = cfg.CORRUPTION.SEVERITY
    if cfg.CORRUPTION.DATASET in {"coloredMNIST", "waterbirds"}:
        biased = True
    # start evaluation
    accs = []
    biased = False
    for i_dom, domain_name in enumerate(dom_names_loop):
        try:
            model.reset()
            logger.info("resetting model")
        except:
            logger.warning("not resetting model")

        for severity in severities:
            log_results = {}
            testset, test_loader = load_dataset(cfg.CORRUPTION.DATASET, cfg.DATA_DIR,
                                                cfg.TEST.BATCH_SIZE,
                                                split='all', domain=domain_name, level=severity,
                                                adaptation=cfg.MODEL.ADAPTATION,
                                                workers=min(cfg.TEST.NUM_WORKERS, os.cpu_count()),
                                                ckpt=os.path.join(cfg.CKPT_DIR, 'Datasets'),
                                                num_aug=cfg.TEST.N_AUGMENTATIONS)

            for epoch in range(cfg.TEST.EPOCH):
                for i, data in enumerate(tqdm(test_loader)):
                    imgs, labels = data[0], data[1]
                    output, return_entropys, filter_ids_1, filter_ids_2, plpd = model([img.to(device) for img in imgs]) if isinstance(imgs, list) else model(imgs.to(device))
                    predictions = output.argmax(1)
                    log_results[f"i"] = {
                        "output": output,
                        "plpd": plpd,
                        "entropys": return_entropys,
                        "filter_ent": filter_ids_1,
                        "filter_plpd": filter_ids_2,
                        "correct": predictions==labels.to(device),
                        "acc": (predictions == labels.to(device)).float().sum() / imgs.shape[0]
                    }
        break
    with open(f"cfg.CORRUPTION.DATASET", "w") as fp:
        json.dump(log_results, fp)
    # return accs
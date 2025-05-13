import os

import timm
from torchvision.models import resnet50, ResNet50_Weights, convnext_base, ConvNeXt_Base_Weights, efficientnet_b0, EfficientNet_B0_Weights
import pickle
# import models.Res as Resnet
from ..models import *
from ..models import Res
from ..models import clip
from ..models.resnet import resnet

def load_model(model_name, checkpoint_dir=None, domain=None, cfg = None):
    if model_name == 'Hendrycks2020AugMix_ResNeXt':
        model = Hendrycks2020AugMixResNeXtNet()
        if checkpoint_dir is not None:
            checkpoint_path = os.path.join(checkpoint_dir, 'Hendrycks2020AugMix_ResNeXt.pt')
            if not os.path.exists(checkpoint_path):
                raise ValueError('No checkpoint found at {}'.format(checkpoint_path))
            model.load_state_dict(torch.load(checkpoint_path))
        else:
            raise ValueError('No checkpoint path provided')
    elif model_name == "resnet50_gn":
        model = timm.create_model("resnet50_gn", pretrained = True)
    # Resnet 50 for ImageNet-C
    elif model_name == 'resnet50':
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    
    elif model_name == 'WideResNet':
        model = WideResNet()
        if checkpoint_dir is not None:
            checkpoint_path = os.path.join(checkpoint_dir, 'WideResNet.pt')
            if not os.path.exists(checkpoint_path):
                raise ValueError('No checkpoint found at {}'.format(checkpoint_path))
            model.load_state_dict(torch.load(checkpoint_path))
        else:
            raise ValueError('No checkpoint path provided')
    
    elif model_name == 'officehome_shot':
        model= OfficeHome_Shot()
        if checkpoint_dir is not None:
            checkpoint_path = os.path.join(checkpoint_dir,'officehome',domain, 'model.pt')
            if not os.path.exists(checkpoint_path):
                raise ValueError('No checkpoint found at {}'.format(checkpoint_path))
            model.load_state_dict(torch.load(checkpoint_path))
    
    elif model_name == 'domainnet126_shot':
        model= DomainNet126_Shot()
        if checkpoint_dir is not None:
            checkpoint_path = os.path.join(checkpoint_dir,'domainnet126',domain, 'model.pt')
            if not os.path.exists(checkpoint_path):
                raise ValueError('No checkpoint found at {}'.format(checkpoint_path))
            model.load_state_dict(torch.load(checkpoint_path))
    
    elif model_name == "resnet18-bn":
        # model = Resnet.resnet18()
        if checkpoint_dir is not None:
            checkpoint_path = os.path.join(checkpoint_dir, "ColoredMNIST_model.pickle")
            with open(checkpoint_path, "rb") as f:
                model = pickle.load(f)
    
    elif model_name == "resnet50-bn":
        if domain == "origin":
            model = Res.resnet50()
            model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=2)
            if checkpoint_dir is not None:
                checkpoint_path = os.path.join(checkpoint_dir, "save.pt")
                model.load_state_dict(torch.load(checkpoint_path))
        else:
            model = resnet("pacs", 50)
            if checkpoint_dir is not None:
                checkpoint_path = os.path.join(checkpoint_dir,'pacs',domain, 'model.pth')
                model.load_state_dict(torch.load(checkpoint_path)["model"])
    elif model_name == 'vit':
        model=timm.create_model('vit_base_patch16_224', pretrained=True)
    elif model_name == 'convnext_base':
        model=convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
    elif model_name == 'efficientnet_b0':
        model=efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    elif model_name == "clip":
        model, _ = clip.load(model_name)
    else:
        raise ValueError('Unknown model name')

    return model

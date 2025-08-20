import torch.nn as nn
from collections import OrderedDict
import copy
from torch.amp import autocast

class FeatureExtractor(nn.Module):
    """
    FeatureExtractor network based on ResNet backbone, we extract features from layer2 and layer3. 
    We reused the trained semanctic encoder ema_encoder.backbone.backbone. 
    Semantic encoder is freeze during training. 
    Feature extractor is trained using cosine similarity loss L_sim 
    """
    def __init__(
        self,
        encoder, 
    ):
        super().__init__()
        resnet_backbone = encoder.backbone.backbone
        modules = list(resnet_backbone.named_children())
        frozen_modules = OrderedDict((f"{name}", module) for name, module in modules[:7])
        train_modules = OrderedDict((f"{name}", copy.deepcopy(module)) for name, module in modules[:7])


        self.frozen_fe = nn.Sequential(frozen_modules)
        self.train_fe = nn.Sequential(train_modules)

        # weights = None

        # try: 
        #     weights = torch.load(ckpt_path, map_location=device)
        #     encoder_weights = {k.replace("ema_encoder.backbone.backbone.", ""): v for k, v in weights["state_dict"].items() if "ema_encoder" in k}
        # except FileNotFoundError:
        #     print(f"Pretrained model not found at {ckpt_path}")

        # if weights is not None: 
        #     self.frozen_fe.load_state_dict(encoder_weights, strict=False)
        #     self.fe = nn.Sequential(used_modules)
        
        self.freeze()
        self.train_fe.requires_grad_(requires_grad=True)

    def freeze(self):
        """
        Free pretrained feature extractors from encoder
        """
        self.frozen_fe.eval()
        self.frozen_fe.requires_grad_(requires_grad=False)
        print("Freeze ema_encoder from DDPM models.")

    def forward(self, x, fe_layers=["layer2", "layer3"]):
        """
        Forward pass that go through both trainable fe and frozen_fe
        """

        out_frozen = []
        out_fe = []

        # Pass through frozen_fe network
        x_frozen = x.clone()
        device = next(self.parameters()).device
        # with autocast(device_type=device.type):
        for name, module in self.frozen_fe.named_children():
            x_frozen = module(x_frozen)

            if name in fe_layers:
                out_frozen.append(x_frozen)

        # Pass through fe network
        x_fe = x.clone()
        # with autocast(device_type=device.type):
        for name, module in self.train_fe.named_children():
            x_fe = module(x_fe)

            if name in fe_layers:
                out_fe.append(x_fe)
        return out_frozen, out_fe

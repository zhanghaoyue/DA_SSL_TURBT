import model.arch.ACMIL as acmil
import utils
import torch
from torch import nn


def acmil_ga(**kwargs):
    conf = type('', (), {})()
    conf.D_inner = kwargs["D_inner"]
    conf.n_token = kwargs["n_token"]
    conf.mask_drop = kwargs["mask_drop"]
    conf.n_masked_patch = kwargs["n_masked_patch"]
    conf.n_class = kwargs["out_channels"]
    conf.D_feat = kwargs["embed_dim"]
    model = acmil.ACMIL_GA(conf)
    # model = acmil_original.ACMIL_GA(conf, D = 128)
    model.name = "ACMIL"
    model.conf = conf

    if kwargs["pretrained_model_path"] is not None:
        path = kwargs['pretrained_model_path']

        checkpoint = torch.load(path)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        
        for k in list(state_dict.keys()):
            # retain only encoder up to before the embedding layer
            if k.startswith('_mil_encoder') and not k.startswith('_mil_encoder.fc') and "classifier" not in k:
                # remove prefix
                state_dict[k[len("_mil_encoder."):]] = state_dict[k]
            del state_dict[k]
        model.load_state_dict(state_dict, strict=False)
        print("pretrained model loaded successfully! <============>")
        print("pretrained model name: %s" % kwargs['pretrained_model_path'])
    return model


def acmil_ga_with_encoder(**kwargs):
    conf = type('', (), {})()
    conf.D_inner = kwargs["D_inner"]
    conf.n_token = kwargs["n_token"]
    conf.mask_drop = kwargs["mask_drop"]
    conf.n_masked_patch = kwargs["n_masked_patch"]
    conf.n_class = kwargs["out_channels"]
    conf.D_feat = kwargs["embed_dim"]

    # Step 1: Initialize ACMIL model
    acmil_model = acmil.ACMIL_GA(conf)
    acmil_model.name = "ACMIL"
    acmil_model.conf = conf

    # Step 2: Create feature encoder
    if kwargs['encoder_type'] == "mlp":
        feature_encoder = MLPFeatureEncoder(
            input_dim=kwargs["embed_dim"], hidden_dim=int(kwargs["embed_dim"]/2),
        )
    elif kwargs['encoder_type'] == "conv1d":
        feature_encoder = Conv1DFeatureEncoder(
            input_dim=kwargs["embed_dim"], hidden_dim=int(kwargs["embed_dim"]/2),
            )

    # Step 3: Load pretrained weights if specified
    if kwargs["pretrained_model_path"] is not None:
        utils.util.load_and_verify_weights(feature_encoder, acmil_model, kwargs["pretrained_model_path"])
    # Step 4: Wrap both into a combined model
    # Step 4: Freeze conv_encoder
    # for param in conv_encoder.parameters(): 
    #     param.requires_grad = False
    full_model = FeatThenACMIL(feature_encoder, acmil_model)
    full_model.name = "ComboACMIL"
    full_model.conf = conf
    setattr(full_model, "_mil_encoder", acmil_model)

    return full_model


def acmil_mha(**kwargs):
    conf = type('', (), {})()
    conf.D_inner = kwargs["D_inner"]
    conf.n_token = kwargs["n_token"]
    conf.mask_drop = kwargs["mask_drop"]
    conf.n_masked_patch = kwargs["n_masked_patch"]
    conf.n_class = kwargs["out_channels"]
    conf.D_feat = kwargs["embed_dim"]
    model = acmil.ACMIL_MHA(conf, n_token=conf.n_token, n_masked_patch=conf.n_token, mask_drop=conf.mask_drop,
                            n_class=kwargs["out_channels"], D_feat=kwargs["embed_dim"])
    model.name = "ACMIL"
    model.conf = conf
    return model


class CoordConv1DEncoder(nn.Module):
    def __init__(self, input_dim=1024, coord_dim=2):
        super(CoordConv1DEncoder, self).__init__()
        self.input_proj = nn.Linear(input_dim + coord_dim, input_dim)
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, int(input_dim/2), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(int(input_dim/2), input_dim, kernel_size=3, padding=1)
        )

    def forward(self, x, coords, mask=None):
        x = torch.cat([x, coords], dim=-1)
        x = self.input_proj(x)
        x = x.transpose(1, 2)
        x = self.encoder(x)
        return x.transpose(1, 2)


class Conv1DFeatureEncoder(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, kernel_size=3):
        super().__init__()
        self.k = kernel_size
        self.conv1 = nn.Conv1d(input_dim,  hidden_dim, kernel_size=self.k, padding=self.k//2)
        self.conv2 = nn.Conv1d(hidden_dim, input_dim,  kernel_size=self.k, padding=self.k//2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, coords=None, mask=None):
        # x: [B, N, D]
        B, N, D = x.shape
        if N == 0:
            # No tokens: return as-is (won’t be used downstream) or create a safe zero output
            return x

        xt = x.transpose(1, 2)  # [B, D, N]

        # If N < k, pad length-dimension up to k, then crop back
        if N < self.k:
            pad_right = self.k - N
            xt = torch.nn.functional.pad(xt, (0, pad_right))     # pad last dim (length)
            y = self.conv1(xt)
            y = self.act(y)
            y = self.conv2(y)
            y = y[..., :N]                     # crop back to original N
        else:
            y = self.conv1(xt)
            y = self.act(y)
            y = self.conv2(y)

        y = y.transpose(1, 2)  # [B, N, D]
        return x + y
    

class MLPFeatureEncoder(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512):
        super(MLPFeatureEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, coords=None, mask=None):
        return x + self.encoder(x)  # Residual connection [B, N, output_dim]


class FeatThenACMIL(nn.Module):
    def __init__(self, feat_encoder, acmil_model):
        super(FeatThenACMIL, self).__init__()
        self.feature_encoder = feat_encoder
        self._mil_encoder = acmil_model

    def forward(self, x, coords, slide_meta, mask=None):
        # x: [B, N, D]
        # ACMIL expects [B, N, D] and mask
        x = self.feature_encoder(x, coords)           # apply conv feature encoder
        return self._mil_encoder(x, mask, slide_meta)   # pass to ACMIL
    
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, et_type, MIL_encoder, feature_dim=1024, proj_dim=128, pred_dim=128):
        """
        dim: feature dimension (default: 1024)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()
        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self._mil_encoder = MIL_encoder
        
        # self.feature_encoder = TransformerEncoderWithCoordinates(
        #     input_dim=feature_dim, 
        #     num_heads=2, 
        #     num_layers=2, 
        #     latent_dim=feature_dim
        # )
        if et_type == 'conv1d':
            self.feature_encoder = Conv1DFeatureEncoder(input_dim=feature_dim, hidden_dim=int(feature_dim/2))
        elif et_type == 'mlp':
            self.feature_encoder = MLPFeatureEncoder(input_dim=feature_dim,hidden_dim=int(feature_dim/2))
        else:
            raise ValueError(f"Unknown encoder type: {et_type}")

        # build a 3-layer projector
        prev_dim = self._mil_encoder.Slide_classifier.fc.weight.shape[1]
        # Store the original fc layer
        original_fc = self._mil_encoder.Slide_classifier.fc
        self._mil_encoder.Slide_classifier.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.InstanceNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.InstanceNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        original_fc,
                                        nn.InstanceNorm1d(feature_dim, affine=False)) # output layer
        self._mil_encoder.Slide_classifier.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.prediction_head = nn.Sequential(nn.Linear(proj_dim, int(proj_dim/2), bias=False),
                                        nn.InstanceNorm1d(int(proj_dim/2)),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(int(proj_dim/2), pred_dim)) # output layer

    def forward(self, x1, x2, coords, mask=None):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
        """
        # Process the feature embeddings through the feature encoder (e.g., transformer)
        z1 = self.feature_encoder(x1, coords, mask)
        z2 = self.feature_encoder(x2, coords, mask)
        
        # Process through MIL-based encoder (e.g., attention mechanism) for further learning
        mil_z1 = self._mil_encoder(z1, mask=mask)[1]
        mil_z2 = self._mil_encoder(z2, mask=mask)[1]

        # Apply prediction head (used during training to encourage agreement)
        p1 = self.prediction_head(mil_z1) # NxC
        p2 = self.prediction_head(mil_z2) # NxC

        return mil_z1, mil_z2, p1, p2, z1, z2
    
    def forward_once(self, x):
        # Single pass (for inference or to get the representation)
        z = self.feature_encoder(x)
        return self.projection_head(z)
        

class TransformerEncoderWithCoordinates(nn.Module):
    def __init__(self, input_dim, num_heads=8, num_layers=4, latent_dim=128):
        super(TransformerEncoderWithCoordinates, self).__init__()
        self.embedding_layer = nn.Linear(input_dim, latent_dim)
        self.positional_encoding = CustomPositionalEncoding(latent_dim)
        self.encoder_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=latent_dim, nhead=num_heads,
                                       dim_feedforward=latent_dim * 4, dropout=0.1),
            num_layers=num_layers
        )

    def forward(self, feature_embeddings, coordinates, mask=None):
        """
        feature_embeddings: [batch_size, num_patches, feature_dim]
        coordinates: [batch_size, num_patches, 2] (x, y coordinates for each patch)
        """
        _, num_patches, _ = feature_embeddings.size()

        # Step 1: Apply feature embedding transformation
        x = self.embedding_layer(feature_embeddings)
        
        # Step 2: Generate and add custom positional encoding from coordinates
        positional_encoding = self.positional_encoding(coordinates, num_patches)
        x = x + positional_encoding
        # Prepare for transformer input
        x = x.transpose(0, 1)  # [max_patches, batch_size, feature_dim]
        if mask is not None:
            # Ensure mask is transposed to match transformer input
            attn_mask = ~mask.bool()  # True for padded tokens
            x = self.encoder_layers(x, src_key_padding_mask=attn_mask)
        else:
            x = self.encoder_layers(x)
        # Transpose back to original shape
        x = x.transpose(0, 1)  # [batch_size, max_patches, feature_dim]
        return x
    

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
    

class Conv1DFeatureEncoder(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, kernel_size=3):
        super(Conv1DFeatureEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, input_dim, kernel_size=kernel_size, padding=1)
        )

    def forward(self, x, coords=None, mask=None):
        # x: [B, N, D] → [B, D, N]
        x_proj = x.transpose(1, 2)
        x_proj = self.encoder(x_proj)
        # [B, D, N] → [B, N, D]
        x_proj = x_proj.transpose(1, 2)
        return x + x_proj
    

class CoordConv1DEncoder(nn.Module):
    def __init__(self, input_dim=1024, coord_dim=2):
        super(CoordConv1DEncoder, self).__init__()
        self.input_proj = nn.Linear(input_dim + coord_dim, input_dim)
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(512, input_dim, kernel_size=3, padding=1)
        )

    def forward(self, x, coords, mask=None):
        x_proj = torch.cat([x, coords], dim=-1)
        x_proj = self.input_proj(x_proj)
        x_proj = x_proj.transpose(1, 2)
        x_proj = self.encoder(x_proj)
        return x_proj.transpose(1, 2) + x


class CoordMLPFeatureEncoder(nn.Module):
    def __init__(self, input_dim=1024, coord_dim=2):
        super(CoordMLPFeatureEncoder, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim + coord_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x, coords, mask=None):
        x_concat = torch.cat([x, coords], dim=-1)  # [B, N, D+2]
        return self.projector(x_concat) + x
    

class CustomPositionalEncoding(nn.Module):
    def __init__(self, latent_dim):
        super(CustomPositionalEncoding, self).__init__()
        self.coordinate_projector = nn.Sequential(
            nn.Linear(2, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, latent_dim)
        )

    def forward(self, coordinates, num_patches):
        """
        Custom positional encoding based on normalized coordinates.
        coordinates: Tensor of shape [batch_size, num_patches, 2] (x, y coordinates)
        num_patches: The actual number of patches in the batch (max_len)
        """
        normalized_coordinates = self.normalize_coordinates(coordinates)

        custom_positional_encoding = self.coordinate_projector(normalized_coordinates)
        
        # Ensure the positional encoding only goes up to the current number of patches
        return custom_positional_encoding[:, :num_patches, :]

    def normalize_coordinates(self, coordinates):
        """
        Normalize the coordinates to range [0, 1].
        """
        # Compute max values for normalization
        batch_max_x = coordinates[:, :, 0].max(dim=1, keepdim=True)[0]
        batch_max_y = coordinates[:, :, 1].max(dim=1, keepdim=True)[0]
        
        # Normalize coordinates within each batch
        normalized_x = coordinates[:, :, 0] / (batch_max_x + 1e-8)
        normalized_y = coordinates[:, :, 1] / (batch_max_y + 1e-8)
        
        return torch.stack([normalized_x, normalized_y], dim=-1)
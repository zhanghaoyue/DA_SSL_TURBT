import math
from torch import nn, Tensor
import torch.nn.functional as F
from arch_util.acmil_utils import *



def pos_enc_1d(D, len_seq):
    if D % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(D))
    pe = torch.zeros(len_seq, D)
    position = torch.arange(0, len_seq).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, D, 2, dtype=torch.float) *
                          -(math.log(10000.0) / D)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MLP_single_layer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP_single_layer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.fc(x)
        return x


class ACMIL_MHA(nn.Module):
    def __init__(self, conf, n_class, D_feat, n_token=1, n_masked_patch=0, mask_drop=0):
        super(ACMIL_MHA, self).__init__()
        self.dimreduction = DimReduction(D_feat, conf.D_inner)
        self.sub_attention = nn.ModuleList()
        for i in range(n_token):
            self.sub_attention.append(
                MutiHeadAttention(conf.D_inner, 8, n_masked_patch=n_masked_patch, mask_drop=mask_drop))
        self.bag_attention = MutiHeadAttention_modify(conf.D_inner, 8)
        self.q = nn.Parameter(torch.zeros((1, n_token, conf.D_inner)))
        nn.init.normal_(self.q, std=1e-6)
        self.n_class = n_class

        self.classifier = nn.ModuleList()
        for i in range(n_token):
            self.classifier.append(Classifier_1fc(conf.D_inner, n_class, 0.0))
        self.n_token = n_token
        self.Slide_classifier = Classifier_1fc(conf.D_inner, n_class, 0.0)

    def forward(self, input):
        input = self.dimreduction(input)
        q = self.q
        k = input
        v = input
        outputs = []
        attns = []
        for i in range(self.n_token):
            feat_i, attn_i = self.sub_attention[i](q[:, i].unsqueeze(0), k, v)
            outputs.append(self.classifier[i](feat_i))
            attns.append(attn_i)

        attns = torch.cat(attns, 1)
        feat_bag = self.bag_attention(v, attns.softmax(dim=-1).mean(1, keepdim=True))

        return torch.cat(outputs, dim=0), self.Slide_classifier(feat_bag), attns


class MHA(nn.Module):
    def __init__(self, conf, n_class, D_feat):
        super(MHA, self).__init__()
        self.dimreduction = DimReduction(D_feat, conf.D_inner)
        self.attention = MutiHeadAttention(conf.D_inner, 8)
        self.q = nn.Parameter(torch.zeros((1, 1, conf.D_inner)))
        nn.init.normal_(self.q, std=1e-6)
        self.n_class = n_class
        self.classifier = Classifier_1fc(conf.D_inner, n_class, 0.0)

    def forward(self, input):
        input = self.dimreduction(input)
        q = self.q
        k = input
        v = input
        feat, attn = self.attention(q, k, v)
        output = self.classifier(feat)

        return output


class MutiHeadAttention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            downsample_rate: int = 1,
            dropout: float = 0.1,
            n_masked_patch: int = 0,
            mask_drop: float = 0.0
    ) -> None:
        super().__init__()
        self.n_masked_patch = n_masked_patch
        self.mask_drop = mask_drop
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.layer_norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)

        if self.n_masked_patch > 0 and self.training:
            # Get the indices of the top-k largest values
            b, h, q, c = attn.shape
            n_masked_patch = min(self.n_masked_patch, c)
            _, indices = torch.topk(attn, n_masked_patch, dim=-1)
            indices = indices.reshape(b * h * q, -1)
            rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:, :int(n_masked_patch * self.mask_drop)]
            masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
            random_mask = torch.ones(b * h * q, c).to(attn.device)
            random_mask.scatter_(-1, masked_indices, 0)
            attn = attn.masked_fill(random_mask.reshape(b, h, q, -1) == 0, -1e9)

        attn_out = attn
        attn = torch.softmax(attn, dim=-1)
        # Get output
        out1 = attn @ v
        out1 = self._recombine_heads(out1)
        out1 = self.out_proj(out1)
        out1 = self.dropout(out1)
        out1 = self.layer_norm(out1)

        return out1[0], attn_out[0]


class MutiHeadAttention_modify(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            downsample_rate: int = 1,
            dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.layer_norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, v: Tensor, attn: Tensor) -> Tensor:
        # Input projections
        v = self.v_proj(v)

        # Separate into heads
        v = self._separate_heads(v, self.num_heads)

        # Get output
        out1 = attn @ v
        out1 = self._recombine_heads(out1)
        out1 = self.out_proj(out1)
        out1 = self.dropout(out1)
        out1 = self.layer_norm(out1)

        return out1[0]


class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x):
        ## x: N x L
        # Reshape x to [batch_size * num_patches, feature_length]
        batch_size, num_patches, feature_length = x.shape
        x_flat = x.view(batch_size * num_patches, feature_length)
        A_V = self.attention_V(x_flat)  # NxD
        A_U = self.attention_U(x_flat)  # NxD
        A = self.attention_weights(A_V * A_U)  # NxK
        # Reshape the attention output to [batch_size, num_patches, K]
        A = A.view(batch_size, num_patches, self.K)  # shape: [batch_size, num_patches, K]
        # Transpose the output to [K, batch_size, num_patches]
        A = A.transpose(0, 1)  # shape: [K, batch_size, num_patches]

        return A  ### K x N


class ABMIL(nn.Module):
    def __init__(self, conf, droprate=0):
        super(ABMIL, self).__init__()
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.attention = Attention_Gated(conf.D_inner, conf.D, 1)
        self.classifier = Classifier_1fc(conf.D_inner, conf.n_class, droprate)

    def forward(self, x, mask=None):  ## x: N x L
        x = x[0]
        med_feat = self.dimreduction(x)
        attn_weights = self.attention(med_feat).squeeze(-1)  # (batch, num_patches)

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))  # Ignore padding

        A_out = attn_weights
        attn_weights = F.softmax(attn_weights, dim=1)  # softmax over N
        afeat = torch.mm(attn_weights, med_feat)  ## K x L
        outputs = self.classifier(afeat)
        return outputs


class ACMIL_GA(nn.Module):
    def __init__(self, conf, D=128, droprate=0):
        super(ACMIL_GA, self).__init__()
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.attention = Attention_Gated(conf.D_inner, D, conf.n_token)
        # One classifier per attention head
        self.classifier = nn.ModuleList([
            Classifier_1fc(conf.D_inner, conf.n_class, droprate)
            for _ in range(conf.n_token)
        ])
        # Fusion: learned feature + 4 meta features
        self.fusion_mlp = nn.Sequential(
            nn.Linear(conf.D_inner + 4, conf.D_inner),
            nn.ReLU(),
            nn.Dropout(droprate)
        )
        self.Slide_classifier = Classifier_1fc(conf.D_inner, conf.n_class, droprate)

        self.n_masked_patch = conf.n_masked_patch
        self.n_token = conf.n_token
        self.mask_drop = conf.mask_drop

    def forward(self, x, coords=None, slide_meta=None, mask=None):  ## x: N x L
        # Ensure x is always [batch, num_patch, feature_dim]
        is_batched = x.dim() == 3
        if not is_batched:
            x = x.unsqueeze(0)  # Add batch dimension [1, num_patch, feature_dim]
            if mask is not None:
                mask = mask.unsqueeze(0)  # Add batch dimension to mask
            if slide_meta is not None:
                slide_meta = slide_meta.unsqueeze(0)
                
        batch_size, num_patches, _ = x.shape

        x = self.dimreduction(x)  # [B, N, D]

        A = self.attention(x) # [N,B,K]
        A = A.permute(1, 0, 2)   # from [N, B, K] to [B, N, K]
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(-1).expand(-1, -1, self.n_token)  # [B, N, K]
            A = A.masked_fill(mask == 0, float('-inf'))

        # Random masking during training
        if self.n_masked_patch > 0 and self.training:
            # Ensure n_masked_patch doesn't exceed the number of patches
            n_masked_patch = min(self.n_masked_patch, num_patches)
            
            topk_vals, topk_indices = torch.topk(A, n_masked_patch, dim=1)  # [B, n_masked, K]
        
            # Randomly select a subset of the masked patches (controlled by mask_drop)
            n_drop = int(self.mask_drop * n_masked_patch)
            if n_drop > 0:
                rand_idx = torch.randperm(n_masked_patch, device=A.device)[:n_drop]

                # Create masking tensor
                drop_mask = torch.ones_like(A, dtype=torch.bool)  # [B, N, K]
                for b in range(batch_size):
                    for k in range(self.n_token):
                        drop_positions = topk_indices[b, rand_idx, k]
                        drop_mask[b, drop_positions, k] = False

                A = A.masked_fill(~drop_mask, -1e9)

        A_softmax = F.softmax(A, dim=1)  # [B, N, K]

        # Aggregate features using attention
        x_trans = x.transpose(1, 2)  # [B, D, N]
        afeat = torch.bmm(x_trans, A_softmax).transpose(1, 2)  # [B, K, D]

        # Classify each attention branch
        branch_outputs = torch.stack([
            self.classifier[k](afeat[:, k])
            for k in range(self.n_token)
        ], dim=0)  # [K, B, C]

        # Slide-level representation using mean pooled attention
        A_slide = F.softmax(A, dim=1).mean(dim=2, keepdim=True)  # [B, N, 1]
        slide_feat = torch.bmm(A_slide.transpose(1, 2), x).squeeze(1)  # [B, D]

        # Feature Fusion
        if slide_meta is not None:
            fused = torch.cat([slide_feat, slide_meta], dim=-1)  # [B, D+4]
            fused = self.fusion_mlp(fused)  # [B, D]
        else:
            fused = slide_feat
        slide_output = self.Slide_classifier(fused)  # [B, C]

        # Remove batch dim if unbatched input
        if not is_batched:
            branch_outputs = branch_outputs.squeeze(1)
            slide_output = slide_output.squeeze(0)
            A = A.squeeze(0)

        return branch_outputs, slide_output, A.unsqueeze(0), fused

    def forward_feature(self, x, use_attention_mask=False):  ## x: N x L
        x = x[0]
        x = self.dimreduction(x)
        A = self.attention(x)  ## K x N

        if self.n_masked_patch > 0 and use_attention_mask:
            # Get the indices of the top-k largest values
            k, n = A.shape
            n_masked_patch = min(self.n_masked_patch, n)
            _, indices = torch.topk(A, n_masked_patch, dim=-1)
            rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:, :int(n_masked_patch * self.mask_drop)]
            masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
            random_mask = torch.ones(k, n).to(A.device)
            random_mask.scatter_(-1, masked_indices, 0)
            A = A.masked_fill(random_mask == 0, -1e9)

        A_out = A
        bag_A = F.softmax(A_out, dim=1).mean(0, keepdim=True)
        bag_feat = torch.mm(bag_A, x)
        return bag_feat



if __name__ == '__main__':
    print('#### Test Case ###')
    import torch
    from types import SimpleNamespace

    # Mock config
    conf = SimpleNamespace(
        D_inner=256,
        n_token=4,
        n_masked_patch=5,
        mask_drop=0.5,
        D_feat=1024,
        n_class=2
    )

    batch_size = 2
    num_patches = 5000

    # Create synthetic data
    x = torch.randn(batch_size, num_patches, conf.D_feat)
    mask = torch.ones(batch_size, num_patches, dtype=torch.bool)  # all valid
    # mask = None
    slide_meta = torch.randn(batch_size, 4)  # 4 slide-level features

    model = ACMIL_GA(conf)

    # Forward pass
    branch_out, slide_out, attn_map = model(x, mask=mask, slide_meta=slide_meta)

    # Assertions
    assert branch_out.shape == (conf.n_token, batch_size, conf.n_class), f"Branch output shape mismatch: {branch_out.shape}"
    assert slide_out.shape == (batch_size, conf.n_class), f"Slide output shape mismatch: {slide_out.shape}"
    assert attn_map.shape == (1, batch_size, num_patches, conf.n_token), f"Attention map shape mismatch: {attn_map.shape}"

    print("✅ ACMIL_GA unit test passed.")

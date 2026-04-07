import torch
import torch.nn as nn
import torch.nn.functional as F

from easydict import EasyDict
from typing import Any, Tuple

from amelia_tf.models.components.self_attention import SelfAttentionBlock
from amelia_tf.models.components.cross_attention import CrossAttentionBlock
from amelia_tf.models.components.gmm import GMM
from amelia_tf.models.components.common import MLP, LayerNorm

class AmeliaTF(nn.Module):
    """ Context-aware model for trajectory prediction on airport data.
    Baseline designed for both, trajectory and context data. Largely based on the SceneTransformer:
    https://arxiv.org/pdf/2106.08417.pdf
    """
    def __init__(self, config: EasyDict) -> None:
        super().__init__()

        self.encoder_config = config.encoder
        self.decoder_config = config.decoder
        
         # configurations for taxiing modes prediction
        self.mode_config = getattr(config, 'mode_predictor', EasyDict({
            'num_modes': 16,           # speed (4) x turning (4)
            'hidden_dim': 64,
            'mode_embed_dim': 32,
            'dropout': 0.1,
            'use_hard_mode': False     # if True, use argmax + straight-through; else use soft weighted embedding
        }))
        
        turn_modes = ['TurnLeft', 'TurnRight', 'Straight', 'Hold']
        speed_modes = ['Accel', 'Decel', 'Normal', 'Hold']
        
        num_turn = len(turn_modes)
        num_speed = len(speed_modes)
        num_modes = num_turn * num_speed
        
        # initialisation
        illegal_mode_mask = torch.zeros(num_modes, dtype=torch.bool)
        
        # traverse all combinations
        for t_idx, t in enumerate(turn_modes):
            for s_idx, s in enumerate(speed_modes):
                mode_idx = t_idx * num_speed + s_idx
                # Only Hold_Hold is llegal, other models inluding Hold are illegal
                if t == 'Hold' and s == 'Hold':
                    illegal_mode_mask[mode_idx] = False
                elif t == 'Hold' or s == 'Hold':
                    illegal_mode_mask[mode_idx] = True
                else:
                    illegal_mode_mask[mode_idx] = False
        
        # update configurations
        self.mode_config.illegal_mode_mask = illegal_mode_mask
        
        print("illegal modes:", torch.nonzero(illegal_mode_mask).flatten().tolist())

        self.in_size = self.encoder_config.in_size + self.encoder_config.interp_flag
        self.embed_size = self.encoder_config.embed_size

        # Agent feature extraction
        # Note: Linear/LayerNorm will operate on the last dimension (Dx). Input x is (B,A,T,Dx).
        self.agents_fe = nn.Sequential(*[
            nn.Linear(self.in_size, self.embed_size),
            LayerNorm(self.embed_size, self.encoder_config.bias),
            nn.ReLU(),
            nn.Linear(self.embed_size, self.embed_size)
        ])

        # Context feature extraction (kept same as before)
        ctx_enc_type = self.encoder_config.context_encoder_type
        if ctx_enc_type == 'v0':
            from amelia_tf.models.components.context import ContextNetv0 as ContextNet
            context_config = self.encoder_config.contextnet_v0
        elif ctx_enc_type == 'v1':
            from amelia_tf.models.components.context import ContextNetv1 as ContextNet
            context_config = self.encoder_config.contextnet_v1
        elif ctx_enc_type == 'v2':
            from amelia_tf.models.components.context import ContextNetv2 as ContextNet
            context_config = self.encoder_config.contextnet_v2
        elif ctx_enc_type == 'v3':
            from amelia_tf.models.components.context import ContextNetv3 as ContextNet
            context_config = self.encoder_config.contextnet_v3
        elif ctx_enc_type == 'v4':
            from amelia_tf.models.components.context import ContextNetv4 as ContextNet
            context_config = self.encoder_config.contextnet_v4
        else:
            raise NotImplementedError
        self.context_fe = ContextNet(context_config)

        # Positional encodings
        self.hist_len = self.encoder_config.hist_len
        self.pred_lens = self.encoder_config.pred_lens
        self.time_pe = nn.Embedding(self.encoder_config.T_size, self.embed_size)

        self.drop = nn.Dropout(self.encoder_config.dropout)

        # Social-Temporal encoding blocks
        # NOTE: mutating encoder_config.in_size so attention blocks know the embed dim.
        # Consider copying the config instead of mutating if that's surprising elsewhere.
        self.encoder_config.in_size = self.embed_size
        self.att_blocks = []
        for _ in range(self.encoder_config.num_satt_pre_blocks):
            self.att_blocks.append(SelfAttentionBlock(self.encoder_config, across='time'))
            self.att_blocks.append(SelfAttentionBlock(self.encoder_config, across='agents'))

        for _ in range(self.encoder_config.num_catt_pre_blocks):
            self.att_blocks.append(CrossAttentionBlock(self.encoder_config))

        for _ in range(self.encoder_config.num_satt_blocks):
            self.att_blocks.append(SelfAttentionBlock(self.encoder_config, across='time'))
            self.att_blocks.append(SelfAttentionBlock(self.encoder_config, across='agents'))

        for _ in range(self.encoder_config.num_catt_blocks):
            self.att_blocks.append(CrossAttentionBlock(self.encoder_config))

        for _ in range(self.encoder_config.num_satt_post_blocks):
            self.att_blocks.append(SelfAttentionBlock(self.encoder_config, across='time'))
            self.att_blocks.append(SelfAttentionBlock(self.encoder_config, across='agents'))
        self.att_blocks = nn.ModuleList(self.att_blocks)

        self.refine_mlp = MLP(self.encoder_config)

        # taxiing modes prediction blocks
        self.mode_classifier = nn.Sequential(
            nn.Linear(self.embed_size, self.mode_config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.mode_config.dropout),
            nn.Linear(self.mode_config.hidden_dim, self.mode_config.num_modes)
        )

        # embedding matrix for modes; we'll use this for both soft (weighted) and hard embeddings.
        self.mode_embedding = nn.Embedding(
            self.mode_config.num_modes,
            self.mode_config.mode_embed_dim
        )

        # fusion of trajectory features and mode embedding
        self.trajectory_fusion = nn.Sequential(
            nn.Linear(self.embed_size + self.mode_config.mode_embed_dim, self.embed_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_size, self.embed_size)
        )

        self.decoder_config.in_size = self.embed_size
        self.decoder_head = GMM(self.decoder_config)

        self.apply(self._init_weights)
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    @property
    def num_dec_heads(self) -> int:
        return self.decoder_head.num_futures

    def get_num_params(self, non_embedding: bool = True) -> int:
        """ Returns the number of parameters in the model. For non-embedding count (default),
        the position embeddings get subtracted.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            if hasattr(self, "time_pe") and hasattr(self.time_pe, "weight"):
                n_params -= self.time_pe.weight.numel()
        return n_params

    def _init_weights(self, module: Any) -> None:
        """ Weight initialization. """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _predict_taxiing_modes(self, trajectory_features):
        """Internal method for predicting taxiing modes.

        Returns:
            mode_logits: (B, A, num_modes)
            mode_embeddings: (B, A, mode_embed_dim)
        """
        B, A, T, D = trajectory_features.shape

        # Use features from the last timestep for mode classification prediction
        last_features = trajectory_features[:, :, -1, :]  # (B, A, D)

        # Classification prediction: output logits for each mode
        mode_logits = self.mode_classifier(last_features)  # (B, A, num_modes)
        
        # ===== Apply illegal mode mask =====
        if hasattr(self.mode_config, "illegal_mode_mask"):
            # illegal_mode_mask: (num_modes,) with True for illegal modes
            illegal_mask = self.mode_config.illegal_mode_mask.to(trajectory_features.device).view(1, 1, -1)  # (1, 1, num_modes)
            mode_logits = mode_logits.masked_fill(illegal_mask, -1e9)

        # Option 1 (default): differentiable soft embedding
        mode_probs = F.softmax(mode_logits, dim=-1)  # (B, A, num_modes)

        # compute weighted sum of embedding matrix: (num_modes, mode_embed_dim)
        emb_matrix = self.mode_embedding.weight  # (num_modes, mode_embed_dim)
        # Mode embeddings (soft): (B, A, mode_embed_dim)
        mode_embeddings_soft = torch.matmul(mode_probs, emb_matrix)

        # If user asked for hard/argmax embeddings, use straight-through estimator:
        if getattr(self.mode_config, "use_hard_mode", False):
            # hard indices and one-hot
            hard_idx = torch.argmax(mode_probs, dim=-1)  # (B, A)
            hard_one_hot = F.one_hot(hard_idx, num_classes=self.mode_config.num_modes).type_as(mode_probs)  # (B, A, num_modes)

            # hard embedding
            mode_embeddings_hard = torch.matmul(hard_one_hot, emb_matrix)  # (B, A, mode_embed_dim)

            # straight-through: use hard in forward but allow gradients from soft
            mode_embeddings = mode_embeddings_hard.detach() + (mode_embeddings_soft - mode_embeddings_soft.detach())
        else:
            mode_embeddings = mode_embeddings_soft

        return mode_logits, mode_embeddings

    def forward(self, x: torch.tensor, **kwargs) -> Tuple:
        """ Model's forward module.

        Inputs
        ------
            x[torch.tensor(B, A, T, D)]: input tensor containing the trajectory information.
                B: batch size
                A: number of agents
                T: trajectory length
                D: number of input dimensions.
            kwargs[Any]: other keyword arguments.
                Should contain a key 'context' containing the map information in vectorized format.
                c[torch.tensor(B, T, P, Dc)]: is the tensor containing the context information
                    P: number of polylines
                    Dc: number of input dimensions of the context.

        Outputs
        -------
            pred_scores[torch.tensor(B, A, T, H)]: prediction scores for each prediction head.
            traj_mu[torch.tensor(B, A, T, H, D)]: predicted means for each trajectory.
            traj_sigma[torch.tensor(B, A, T, H, D)]: predicted sigmas for each trajectory.
            mode_logits[torch.tensor(B, A, num_modes)]: predicted taxiing mode logits.
        """
        device = x.device
        B, A, T, D = x.size()
        assert T <= self.encoder_config.T_size, \
            f"Can't forward sequence of length {T}, time block size is {self.encoder_config.T_size}"

        # x: (B, A, T, Dx) -> pass through agents feature extractor (operates on last dim)
        x = self.agents_fe(x)

        # time positional embedding
        time_pe = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, T)
        time_emb = self.time_pe(time_pe).unsqueeze(dim=0)  # (1, 1, T, embed)
        x = self.drop(x + time_emb)  # broadcasting across B and A

        # context and adjacency
        cx = kwargs.get('context')
        adj = kwargs.get('adjacency')
        assert cx is not None, "context (cx) must be provided in kwargs"

        # Context encoder - assumed to return features in expected shape for CrossAttentionBlock(s).
        # Original code comment suggested cx -> (B * A, D=embed_size) but block implementations decide.
        cx = self.context_fe(cx, adj=adj)

        # transformer blocks (time + agent + cross-attention)
        mask = kwargs.get('mask')
        for block in self.att_blocks:
            x = block(x, cx, mask=mask)

        # prediction refinement
        x = self.refine_mlp(x)

        # taxiing modes decoding (returns logits and embeddings)
        mode_logits, mode_embeddings = self._predict_taxiing_modes(x)
        mode_probs = F.softmax(mode_logits, dim=-1)  # (B, A, num_modes)
        
        # ---- mode-conditioned trajectory decoder ----
        num_modes = self.mode_config.num_modes
        H = self.decoder_head.num_futures  # number of heads per mode
        
        # Expand x across modes
        x_exp = x.unsqueeze(2).expand(-1, -1, num_modes, -1, -1)  # (B,A,M,T,embed)
        
        mode_emb_table = self.mode_embedding.weight  # (num_modes, embed_mode)
        mode_emb = mode_emb_table.view(1,1,num_modes,1,-1).expand(B, A, num_modes, T, -1)
        
        decoder_input = torch.cat([x_exp, mode_emb], dim=-1)  # (B,A,M,T,embed+mode_embed)
        decoder_input = self.trajectory_fusion(decoder_input)
        # print(decoder_input.size())
        decoder_input = x_exp + decoder_input
        # GMM decoder
        traj_mu, traj_sigma = self.decoder_head(decoder_input)
        
        # reshape back
        traj_mu = traj_mu.view(B, A, num_modes, T, 3, H).permute(0,1,3,4,2,5)
        traj_sigma = traj_sigma.view(B, A, num_modes, T, 3, H).permute(0,1,3,4,2,5)
        
        return mode_probs, traj_mu, traj_sigma

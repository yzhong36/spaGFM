import torch.nn as nn
import copy
import torch
import torch.nn.functional as F
import random
from spaGFM.stRoamer.models.encoder import Transformer_Encoder
from spaGFM.stRoamer.utils.training_func import mask_feature, mask_patterns

class stRoamer(nn.Module):
    """
    spaGFM model for self-supervised graph representation learning.
    
    This model uses a masked autoencoder approach with momentum contrast to learn
    robust node and subgraph representations from graph data.
    """
    def __init__(self, params):
        """
        Initialize the spaGFM model.
        
        Args:
            params (dict): Model configuration parameters including:
                - mode: 'pretrain', 'adaptation', or 'inference'
                - n_total_rw: Total number of random walks
                - n_visible_rw: Number of visible (unmasked) random walks
                - mask_feature_ratio: Ratio of features to mask
                - mask_node_ratio: Ratio of nodes to mask in patterns
                - input_dim: Input feature dimension
                - hidden_dim: Hidden dimension for encoders
                - rw_encoder: Parameters for random walk encoder
                - subgraph_encoder: Parameters for subgraph encoder
                - subgraph_decoder: Parameters for subgraph decoder
                - variance_loss: Whether to use variance regularization
                - variance_target: Target variance for regularization
        """
        super(stRoamer, self).__init__()
        self.mode = params["mode"]
        self.total_rw_count = params['n_total_rw']
        self.visible_rw_count = params['n_visible_rw']
        self.masked_rw_count = self.total_rw_count - self.visible_rw_count
        self.mask_feature_ratio = params['mask_feature_ratio']
        self.mask_node_ratio = params['mask_node_ratio']

        self.rw_encoder_params = params["rw_encoder"]
        self.subgraph_encoder_params = params["subgraph_encoder"]

        self.rw_encoder_params['input_dim'] = params['input_dim']
        self.rw_encoder_params["pre_projection"] = True
        self.rw_encoder_params["duo_embedding"] = True
        self.rw_encoder_params["simple_transformer"] = True
        self.rw_encoder_params["hidden_dim"] = params["hidden_dim"]

        self.subgraph_encoder_params["input_dim"] = self.rw_encoder_params["hidden_dim"]
        self.subgraph_encoder_params["pre_projection"] = False
        self.subgraph_encoder_params["duo_embedding"] = False
        self.subgraph_encoder_params["simple_transformer"] = False
        self.subgraph_encoder_params["hidden_dim"] = params["hidden_dim"]

        self.rw_encoder = Transformer_Encoder(self.rw_encoder_params)
        self.subgraph_encoder = Transformer_Encoder(self.subgraph_encoder_params)

        ######## Student-Teacher MAE style
        self.subgraph_decoder_params = params["subgraph_decoder"]

        self.subgraph_decoder_params["input_dim"] = self.subgraph_encoder_params["hidden_dim"]
        self.subgraph_decoder_params["pre_projection"] = False
        self.subgraph_decoder_params["duo_embedding"] = False
        self.subgraph_decoder_params["simple_transformer"] = False
        self.subgraph_decoder_params["hidden_dim"] = params["hidden_dim"]

        self.subgraph_decoder = Transformer_Encoder(self.subgraph_decoder_params)
        self.linear_decoder = nn.Linear(self.subgraph_decoder_params["hidden_dim"], self.subgraph_decoder_params["hidden_dim"])

        self.mask_token = nn.Parameter(torch.zeros(1, self.subgraph_decoder_params["hidden_dim"]), requires_grad=True)
        nn.init.normal_(self.mask_token, std=0.02)

        self.online_rw_encoder = copy.deepcopy(self.rw_encoder)
        self.online_subgraph_encoder = copy.deepcopy(self.subgraph_encoder)
        for module in [self.online_rw_encoder, self.online_subgraph_encoder]:
            for param in module.parameters():
                param.requires_grad = False
        ########

        self.variance_loss = params["variance_loss"]
        self.variance_target = params["variance_target"]
        self.covariance_loss = params["covariance_loss"]
        self.covariance_weight = params["covariance_weight"]
        self.loss_fn = params.get("loss_fn", "l2")

    def ema_update(self, alpha=0.99):
        """
        Update online encoders using exponential moving average.
        
        Args:
            alpha (float): EMA decay factor
        """
        with torch.no_grad():
            # Perform EMA update in fp32 to prevent precision loss with small (1-alpha) values
            for online_param, param in zip(self.online_rw_encoder.parameters(), self.rw_encoder.parameters()):
                online_param.data.copy_(
                    online_param.data.float().mul_(alpha).add_(param.data.float(), alpha=1 - alpha).to(online_param.dtype)
                )

            for online_param, param in zip(self.online_subgraph_encoder.parameters(), self.subgraph_encoder.parameters()):
                online_param.data.copy_(
                    online_param.data.float().mul_(alpha).add_(param.data.float(), alpha=1 - alpha).to(online_param.dtype)
                )

    def subgraph_augmentation(self, graph, nodes):
        """
        Apply data augmentation to graph and nodes.
        
        Args:
            graph: PyG Data object
            nodes: Random walk node sequences
            
        Returns:
            tuple: Augmented graph and nodes
        """
        feat_mode = random.choice(['col', 'row'])
        pattern_mode = random.choice(['mask', 'random'])

        feat = torch.cat([graph.x, torch.zeros(1, graph.x.size(1), device=graph.x.device)], dim=0)  # Add a dummy node for masking
        feat, _ = mask_feature(feat, self.mask_feature_ratio, mode=feat_mode, training=True)

        nodes, _ = mask_patterns(nodes, self.mask_node_ratio, mode=pattern_mode, training=True)

        return feat, nodes

    def feature_gather(self, feat, nodes):
        num_patterns, num_nodes, pattern_length = nodes.shape
        nodes_flat = nodes.view(-1)  # Shape: [num_patterns * num_nodes * pattern_length]
        feat_gathered = feat[nodes_flat]  # Shape: [num_patterns * num_nodes * pattern_length, d]
        feat_gathered = feat_gathered.view(num_patterns * num_nodes, pattern_length, -1)  # Reshape to [num_patterns * num_nodes, pattern_length, d]

        return feat_gathered

    def pretrain(self, graph, nodes):
        """
        Perform self-supervised pretraining with masking and reconstruction.
        
        Args:
            graph: PyG Data object
            nodes: Random walk sequences [num_patterns, num_nodes, walk_length]
            
        Returns:
            tuple: (loss, subgraph_embeddings)
        """
        visible_rw_count = self.visible_rw_count
        masked_rw_count = self.masked_rw_count

        num_patterns, num_nodes, _ = nodes.shape
        shuffle_idx = torch.randperm(num_patterns, device=nodes.device)
        unshuffle_idx = torch.argsort(shuffle_idx)
        mask = torch.zeros(num_patterns, dtype=torch.bool, device=nodes.device)
        mask[shuffle_idx[visible_rw_count:]] = True

        ## teacher encoding
        with torch.no_grad():
            _, online_rw_feat = self.online_rw_encoder(self.feature_gather(graph.x, nodes))
            online_rw_feat = online_rw_feat.view(num_patterns, num_nodes, self.rw_encoder_params["hidden_dim"])
            online_target = self.online_subgraph_encoder(torch.transpose(online_rw_feat, 0, 1)).detach()

        ## student encoding
        feat_aug, nodes_aug = self.subgraph_augmentation(graph, nodes)
        _, rw_feat = self.rw_encoder(self.feature_gather(feat_aug, nodes_aug[shuffle_idx[:visible_rw_count]]))
        rw_feat = rw_feat.view(visible_rw_count, num_nodes, self.rw_encoder_params["hidden_dim"])
        mask_tokens = torch.transpose(self.mask_token.repeat(masked_rw_count, rw_feat.size(1), 1), 0, 1)

        subgraph_emb = self.subgraph_encoder(torch.transpose(rw_feat, 0, 1))
        subgraph_emb_with_mask = torch.cat([subgraph_emb, mask_tokens], dim=1)
        subgraph_emb_with_mask = subgraph_emb_with_mask[:,unshuffle_idx,:]
        recon_subgraph_emb = self.subgraph_decoder(subgraph_emb_with_mask)
        pred = self.linear_decoder(recon_subgraph_emb)  

        ## Objective loss
        loss_fn = F.mse_loss if self.loss_fn == "l2" else F.l1_loss
        loss = loss_fn(pred[:, mask, :], online_target[:, mask, :])

        # Upcast to fp32 before mean/variance to prevent numerical instability in bf16/float16
        subgraph_instance_emb = subgraph_emb.float().mean(dim=1)
        if self.variance_loss:
            subgraph_emb_std = torch.sqrt(torch.var(subgraph_instance_emb, dim=0) + 1e-4)
            var_loss = torch.mean(F.relu(self.variance_target - subgraph_emb_std))
            loss = loss + var_loss

            if self.covariance_loss:
                n = subgraph_instance_emb.size(0)
                d = subgraph_instance_emb.size(1)

                subgraph_centered = subgraph_instance_emb - subgraph_instance_emb.mean(dim=0)
                cov_subgraph_agg = (subgraph_centered.T @ subgraph_centered) / (n - 1)
                cov_loss = (cov_subgraph_agg.pow(2).sum() - torch.diagonal(cov_subgraph_agg).pow(2).sum()) / d
                
                loss = loss + cov_loss * self.covariance_weight

        return loss, subgraph_instance_emb

    def inference(self, graph, nodes):
        """
        Extract embeddings for inference/downstream tasks.
        
        Args:
            graph: PyG Data object
            nodes: Random walk sequences
            
        Returns:
            tuple: (start_embeddings, subgraph_embeddings)
        """
        num_patterns, num_nodes, _ = nodes.shape
        start_emb, rw_feat = self.rw_encoder(self.feature_gather(graph.x, nodes))

        start_emb = start_emb.view(num_patterns, num_nodes, self.rw_encoder_params["hidden_dim"])
        # Upcast to fp32 for stable mean calculation in bf16/float16
        start_emb_instance_emb = start_emb.float().mean(dim=0)

        rw_feat = rw_feat.view(num_patterns, num_nodes, self.rw_encoder_params["hidden_dim"])
        subgraph_emb = self.subgraph_encoder(torch.transpose(rw_feat, 0, 1))
        # Upcast to fp32 for stable mean calculation in bf16/float16
        subgraph_instance_emb = subgraph_emb.float().mean(dim=1)

        return start_emb_instance_emb, subgraph_instance_emb

    def forward(self, graph, nodes):
        if self.mode in {'pretrain', 'adaptation'}:
            return self.pretrain(graph, nodes)
        elif self.mode == 'inference':
            return self.inference(graph, nodes)
        else:
            raise ValueError("Invalid mode")
        

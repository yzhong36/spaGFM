from spaGFM.stRoamer.models.stRoamer import stRoamer
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class stRoamer_ft(stRoamer):
    def __init__(self, model_file, device, ft_params, *args, **kwargs):
        
        logger.info(f'Loading model parameters from {model_file}')
        model_dict = torch.load(model_file, map_location=device, weights_only=True)
        model_params = model_dict['model_state']
        
        self.pretrain_params = model_dict['params']
        self.ft_params = ft_params

        super().__init__(self.pretrain_params, *args, **kwargs)

        if self.ft_params['subgraph_cls_token']:
            self.cls_token = nn.Parameter(
                torch.zeros(1, 1, self.pretrain_params['hidden_dim']),
                requires_grad=not self.ft_params['frozen_backbone'],
            )
            nn.init.normal_(self.cls_token, std=0.02)

        # Pooling over an (unordered) set of pattern tokens.
        # Backward-compatible default: if subgraph_cls_token is enabled (and backbone is trainable), use CLS;
        # otherwise mean pooling.
        self.subgraph_pooling = self.ft_params.get('subgraph_pooling')
        if self.subgraph_pooling is None:
            self.subgraph_pooling = 'cls' if (self.ft_params['subgraph_cls_token'] and not self.ft_params['frozen_backbone']) else 'mean'

        if self.subgraph_pooling == 'attn':
            self.pool_attn = nn.Linear(self.pretrain_params['hidden_dim'], 1, bias=False)
        
        if self.ft_params['linear_decoder'] == True:
            self.mlp_decoder = nn.Linear(self.pretrain_params['hidden_dim'], self.ft_params['output_dim'])
        else:
            self.mlp_decoder = nn.Sequential(
                nn.Linear(self.pretrain_params['hidden_dim'], self.pretrain_params['hidden_dim']),
                nn.ReLU(),
                nn.Linear(self.pretrain_params['hidden_dim'], self.ft_params['output_dim'])
            )

        if self.ft_params['frozen_backbone']:
            for param in self.rw_encoder.parameters():
                param.requires_grad = False
            for param in self.subgraph_encoder.parameters():
                param.requires_grad = False

            self.rw_encoder.eval()
            self.subgraph_encoder.eval()
        else:
            for param in self.rw_encoder.parameters():
                param.requires_grad = False
            for param in self.subgraph_encoder.parameters():
                param.requires_grad = True

            self.rw_encoder.eval()
            self.subgraph_encoder.train()

        self.load_state_dict(model_params, strict=False if self.ft_params['mode'] == 'train' else True)
        self.to(device)

    def forward(self, graph, nodes):
        num_patterns, num_nodes, _ = nodes.shape

        _, rw_feat = self.rw_encoder(self.feature_gather(graph.x, nodes))

        rw_feat = rw_feat.view(num_patterns, num_nodes, self.rw_encoder_params["hidden_dim"])
        rw_feat = torch.transpose(rw_feat, 0, 1)

        use_cls_token = (
            self.subgraph_pooling == 'cls'
            and self.ft_params['subgraph_cls_token']
            and not self.ft_params['frozen_backbone']
        )
        if use_cls_token:
            cls_token = self.cls_token.expand(num_nodes, 1, -1)
            rw_feat = torch.cat([cls_token, rw_feat], dim=1)

        subgraph_tokens = self.subgraph_encoder(rw_feat)

        if self.subgraph_pooling == 'cls' and use_cls_token:
            subgraph_emb = subgraph_tokens[:, 0, :]
        else:
            # For mean/attn pooling, pool only over pattern tokens.
            tokens_to_pool = subgraph_tokens[:, 1:, :] if use_cls_token else subgraph_tokens
            if self.subgraph_pooling == 'attn':
                attn_logits = self.pool_attn(tokens_to_pool).squeeze(-1)
                attn_weights = torch.softmax(attn_logits, dim=1).unsqueeze(-1)
                subgraph_emb = (tokens_to_pool * attn_weights).sum(dim=1)
                return attn_weights, self.mlp_decoder(subgraph_emb)
            else:
                subgraph_emb = tokens_to_pool.mean(dim=1)

        return self.mlp_decoder(subgraph_emb)

from typing import Any, Union
import torch
import torch.nn as nn


class Transformer_Encoder(nn.Module):
    def __init__(self, params: dict[str, Any]) -> None:
        super(Transformer_Encoder, self).__init__()
        self.input_dim: int = params['input_dim']
        self.hidden_dim: int = params['hidden_dim']
        self.num_heads: int = params['n_heads']
        self.num_layers: int = params['n_layers']
        self.dropout: float = params['dropout']

        self.pre_projection = nn.Linear(self.input_dim, self.hidden_dim) if params['pre_projection'] else None
        self.duo_embedding = params['duo_embedding']
        self.simple_transformer = params['simple_transformer']

        if self.simple_transformer:
            self._init_simple_transformer_encoder()
        else:
            self._init_transformer_encoder()

    def _init_simple_transformer_encoder(self):
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.num_heads,
                dim_feedforward=self.hidden_dim * 4,
                dropout=self.dropout,
                batch_first=True,
                norm_first=False
            ),
            num_layers=self.num_layers
        )

    def _init_transformer_encoder(self):
        self.encoder = nn.ModuleList([nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            batch_first=True,
            norm_first=False
        ) for _ in range(self.num_layers)])

        self.encoder_norm = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)])

    def forward(self, x: torch.Tensor, *_args, **_kwargs) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the transformer encoder.
        
        Args:
            x: Input tensor of shape [batch, seq_len, input_dim]
            
        Returns:
            If duo_embedding is True: tuple of (start_emb, agg_emb)
            Otherwise: encoded tensor of shape [batch, seq_len, hidden_dim]
        """
        if self.pre_projection is not None:
            x = self.pre_projection(x)

        if self.simple_transformer:
            x = self.encoder(x)
        else:
            for layer, norm in zip(self.encoder, self.encoder_norm):
                last_x = x
                x = layer(norm(x))
                x = x + last_x  # Residual connection

        if self.duo_embedding:
            start_emb = x[:, 0, :]
            # Upcast to fp32 for stable mean calculation in bf16/float16
            agg_emb = x.float().mean(dim=1)
            return start_emb, agg_emb
        else:
            return x

        
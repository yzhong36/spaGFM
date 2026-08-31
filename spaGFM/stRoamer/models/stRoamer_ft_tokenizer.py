import logging
import torch
from spaGFM.stRoamer.models.stRoamer_ft import stRoamer_ft

logger = logging.getLogger(__name__)


class stRoamer_ft_tokenizer(stRoamer_ft):
    def forward(self, graph, nodes):
        num_patterns, num_nodes, _ = nodes.shape

        _, rw_feat = self.rw_encoder(self.feature_gather(graph.x, nodes))

        rw_feat = rw_feat.view(num_patterns, num_nodes, self.rw_encoder_params["hidden_dim"])
        rw_feat = torch.transpose(rw_feat, 0, 1)

        return rw_feat

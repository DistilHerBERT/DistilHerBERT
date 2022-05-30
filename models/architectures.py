from torch import nn
from transformers.activations import  GELUActivation
import bert


class Config():
    pass


class BertAgNews(nn.Module):
    def __init__(self):
        self.config_dict = {
            'num_attention_heads' : 12,
            'config.hidden_size' : 768,
            'config.attention_probs_dropout_prob' : 0.1,

            'config.layer_norm_eps' : 1e-12,
            'config.hidden_dropout_prob' : 0.1,

            'config.intermediate_size' : 3072,
            'config.hidden_act' : GELUActivation(),
            'config.num_hidden_layers' : 12,
            'vocab_size': 50000,
            'pad_token_id': 1,
            'max_position_embeddings': 514,
            'type_vocab_size': 2,
        }
        self.config = Config()
        for key, val in self.config_dict.items():
            self.config.__setattr__(key, val)
from torch import nn
from transformers.activations import  GELUActivation
import bert
from typing import Optional, Union, Tuple
import torch


class Config():
    pass


class BertAgNews(nn.Module):
    def __init__(self):


class BertAgNews(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config_dict = {
            'num_attention_heads': 12,
            'hidden_size': 768,
            'attention_probs_dropout_prob': 0.1,

            'layer_norm_eps': 1e-12,
            'hidden_dropout_prob': 0.1,

            'intermediate_size': 3072,
            'hidden_act': GELUActivation(),
            'num_hidden_layers': 12,
            'vocab_size': 50000,
            'pad_token_id': 1,
            'max_position_embeddings': 514,
            'type_vocab_size': 2,
            'classifier_dropout': 0.1,
            'num_labels': 4,
        }
        self.config = Config()
        for key, val in self.config_dict.items():
            self.config.__setattr__(key, val)


        self.num_labels = config.num_labels
        self.config = config

        self.bert = bert.BertModel(config)
        classifier_dropout = config.classifier_dropout

        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[torch.Tensor]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        assert not return_dict
        return_dict = False

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        output = (logits,) + outputs[2:]

        return output

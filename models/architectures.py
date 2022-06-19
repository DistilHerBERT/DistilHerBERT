from torch import nn
from transformers.activations import ACT2FN

from . import bert
from typing import Optional, Tuple
import torch


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    # taken from https://github.com/huggingface/transformers/blob/39b4aba54d349f35e2f0bd4addbe21847d037e9e/src/transformers/models/bert/modeling_bert.py#L701
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertLMPredictionHead(nn.Module):
    # taken from https://github.com/huggingface/transformers/blob/39b4aba54d349f35e2f0bd4addbe21847d037e9e/src/transformers/models/bert/modeling_bert.py#L681
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))  # TODO: that bias is init to zero...

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertForMLM(nn.Module):
    # taken from https://github.com/huggingface/transformers/blob/39b4aba54d349f35e2f0bd4addbe21847d037e9e/src/transformers/models/bert/modeling_bert.py#L1292
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = bert.BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        # self.dropout = nn.Dropout(0.1)

    def forward(
            self,
            input_ids: Optional[torch.Tensor],
            attention_mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor]:
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
        )

        # pooled_output = outputs[1]
        # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)
        # output = logits

        prediction_scores = self.cls(outputs[0])

        return prediction_scores


class BertAgNews(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_labels = 4
        self.config = config
        self.bert = bert.BertModel(config)

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

    def forward(
            self,
            input_ids: Optional[torch.Tensor],
            attention_mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor]:
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        output = logits

        return output

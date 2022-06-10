from torch import nn
from . import bert
from typing import Optional, Union, Tuple
import torch


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

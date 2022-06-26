from torch import nn

from models import bert


class BertPolemo(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.config = config
        self.bert = bert.BertModel(config)
        self.ner_classifier = nn.Sequential(nn.Dropout(0.3, inplace=False),
                                            nn.Linear(config.hidden_size, self.num_labels),
                                            )

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.ner_classifier(pooled_output)
        return output
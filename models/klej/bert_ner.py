from torch import nn

from models import bert


class BertNER(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.config = config
        self.bert = bert.BertModel(config)
        self.ner_classifier = nn.Sequential(nn.Dropout(0.1, inplace=False),
                                            nn.Linear(config.hidden_size, self.num_labels),
                                            )

    def forward(self, input_ids, attention_mask):
        x = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = x[1]
        output = self.ner_classifier(pooled_output)
        return output

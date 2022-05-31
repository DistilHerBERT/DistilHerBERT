from models.bert import BertModel
from transformers import AutoTokenizer, AutoModel


tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
model = AutoModel.from_pretrained("allegro/herbert-base-cased")
print(model)


#tmodel is exact copy of the bert model, but is implemented fully in torch
tmodel = BertModel(model.config)
tmodel.load_state_dict(model.state_dict())

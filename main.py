from models.bert import BertModel
from transformers import AutoTokenizer, AutoModel
import torch


tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
model = AutoModel.from_pretrained("allegro/herbert-base-cased")


#tmodel is exact copy of the bert model, but is implemented fully in torch
tmodel = BertModel(model.config)
tmodel.load_state_dict(model.state_dict())


tokenized = tokenizer.batch_encode_plus(
        [
            (
                "A potem szedł środkiem drogi w kurzawie, bo zamiatał nogami, ślepy dziad prowadzony przez tłustego kundla na sznurku.",
                "A potem leciał od lasu chłopak z butelką, ale ten ujrzawszy księdza przy drodze okrążył go z dala i biegł na przełaj pól do karczmy."
            )
        ],
    padding='longest',
    add_special_tokens=True,
    return_tensors='pt'
    )


model.eval()
tmodel.eval()

torch.manual_seed(0)
output = model(**tokenized)
torch.manual_seed(0)
toutput = tmodel(**tokenized)

dif1 = output[0] - toutput[0]
dif2 = output[1] - toutput[1]
print("Checking if output from both models are equal")
print("all values should be zero:")
print(dif1.abs().max().item(), dif2.abs().max().item())

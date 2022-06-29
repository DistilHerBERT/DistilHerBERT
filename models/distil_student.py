from itertools import compress, cycle
from torch.nn import Identity, LayerNorm
from transformers import AutoModel, BertConfig, AutoTokenizer
from transformers.models.bert.modeling_bert import BertModel as BERT
from transformers.models.bert.modeling_bert import BertEncoder as BertEncoderHF

from models.bert import BertModel, BertEncoder


def yield_every_other(g):
    return compress(g, cycle([True, False]))


def copy_weights_from_teacher_to_student(teacher, student):
    if isinstance(teacher, BertModel) or isinstance(teacher, BERT):
        for teacher_part, student_part in zip(teacher.children(), student.children()):
            copy_weights_from_teacher_to_student(teacher_part, student_part)
    elif isinstance(teacher, BertEncoder) or isinstance(teacher, BertEncoderHF):
        teacher_encoding_layers = list(yield_every_other(teacher.layer.children()))
        student_encoding_layers = [layer for layer in next(student.children())]
        for i in range(len(student_encoding_layers)):
            student_encoding_layers[i].load_state_dict(teacher_encoding_layers[i].state_dict())
    else:
        student.load_state_dict(teacher.state_dict())


def creat_student(teacher_model=None):
    if teacher_model is None:
        teacher_model = AutoModel.from_pretrained("allegro/herbert-base-cased")
    else:
        import copy
        teacher_model = copy.deepcopy(teacher_model)

    teacher_config = teacher_model.config.to_dict()
    teacher_config['num_hidden_layers'] //= 2
    student_model = BertModel(BertConfig.from_dict(teacher_config), add_pooling_layer=False)
    copy_weights_from_teacher_to_student(teacher_model, student_model)
    student_model.embeddings.token_type_embeddings = Identity()
    #print(student_model)
    return student_model


def test(model):
    tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")
    output = model(
        **tokenizer.batch_encode_plus(
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
    )
    return output


if __name__ == "__main__":
    student = creat_student()
    output = test(student)

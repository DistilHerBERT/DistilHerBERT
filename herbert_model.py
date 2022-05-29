from transformers import AutoTokenizer, AutoModel


def main():
    tokenizer = AutoTokenizer.from_pretrained("allegro/herbert-base-cased")

    model = AutoModel.from_pretrained("allegro/herbert-base-cased")
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


if __name__ == '__main__':
    main()

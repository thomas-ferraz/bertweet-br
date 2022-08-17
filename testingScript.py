from transformers import pipeline
from transformers import AutoTokenizer  # Or BertTokenizer

tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=False)

fill_mask = pipeline(
    "fill-mask",
    model="./alternative_training/release",
    tokenizer=tokenizer
)

print(fill_mask("O [MASK] é um grande problema atualmente."))

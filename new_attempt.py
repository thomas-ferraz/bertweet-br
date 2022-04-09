from transformers import AutoTokenizer  # Or BertTokenizer
from transformers import AutoModelForPreTraining  # Or BertForPreTraining for loading pretraining heads
from transformers import AutoModel  # or BertModel, for BERT without pretraining heads
import torch
from transformers import AutoConfig, AutoModelForMaskedLM

from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

paths = [str(x) for x in Path("./data/text/tweet_text/").glob("**/*.txt")]

# # Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# # Customize training
# tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
#     "<s>",
#     "<pad>",
#     "</s>",
#     "<unk>",
#     "<mask>",
# ])

# tokenizer.save_model("test_error")

from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing


tokenizer = ByteLevelBPETokenizer(
    "./test_error/vocab.json",
    "./test_error/merges.txt",
)

tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)

print(tokenizer.encode("Bom dia, seus miseráveis.").tokens)

# Check that PyTorch sees it
import torch
torch.cuda.is_available()

config = AutoConfig.from_pretrained('neuralmind/bert-base-portuguese-cased')

model = AutoModelForMaskedLM.from_pretrained('neuralmind/bert-base-portuguese-cased')

model.num_parameters()

tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')

from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./data/text/tweet_text/text_0.txt",
    block_size=128,
)

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./test_error",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

trainer.save_model("./test_error")

from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="./test_error",
    tokenizer=tokenizer
)

print(fill_mask("Esse é o começo de uma bela [MASK]"))

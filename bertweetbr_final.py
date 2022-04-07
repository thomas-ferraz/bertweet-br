from transformers import AutoTokenizer  # Or BertTokenizer
from transformers import AutoModelForPreTraining  # Or BertForPreTraining for loading pretraining heads
from transformers import AutoModel  # or BertModel, for BERT without pretraining heads
import torch
from transformers import AutoConfig, AutoModelForMaskedLM

print("Parte 0")

# Aqui teria o tokenizer manual, mas vamos usar o do bertimbau por enquanto

print("Parte 1")

#modelo do BERTweet aqui, mas estava dando erro entao usei o Roberta
model = AutoModelForPreTraining.from_pretrained('neuralmind/bert-base-portuguese-cased')

tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=False)

#tokens = tokenizer('bom dia como estão?')
#print(tokens)

# Qual seria o config para o bertimbau?
from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

# Qual seria o MLM do bertimbau?
from transformers import RobertaForMaskedLM

model = RobertaForMaskedLM(config=config)

config = AutoConfig.from_pretrained("neuralmind/bert-base-portuguese-cased")

model = AutoModelForMaskedLM.from_config(config)

print(model.num_parameters())

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

print("Parte 2")

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./BERTweetBR",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=16,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

print("Parte 3")

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

trainer.save_model("./BERTweetBR")

print("Parte 4")

from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="./BERTweetBR",
    tokenizer=tokenizer
)

print(fill_mask("Esse é o começo de uma bela [MASK]"))


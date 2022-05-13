text = "É um belo [MASK]."
model_checkpoint = 'neuralmind/bert-base-portuguese-cased'
chunk_size = 128
train_size = 1_000
test_size = int(0.1 * train_size)

# Pega o model
from transformers import AutoModelForMaskedLM
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

# Pega o tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Teste base
import torch
inputs = tokenizer(text, return_tensors="pt")
token_logits = model(**inputs).logits
# Find the location of [MASK] and extract its logits
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
mask_token_logits = token_logits[0, mask_token_index, :]
# Pick the [MASK] candidates with the highest logits
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

for token in top_5_tokens:
    print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")

# Prepara datasets
from datasets import load_dataset
imdb_dataset = load_dataset('text', data_files={'train': ['./data/text/tweet_text/quick.txt'], 'test': './data/text/tweet_text/quick2.txt'})

# Funcao para tokenizacao
def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

# Tokenizando datasets
tokenized_datasets = imdb_dataset.map(
    tokenize_function, batched=True, remove_columns=["text"]
)
print(tokenized_datasets)

# Funcao para agrupar textos por chunk
def group_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result

# Aplicando group_texts para dataset tokenizado
lm_datasets = tokenized_datasets.map(group_texts, batched=True)
print(lm_datasets)

# Prepara Data Collator
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

# Diminuir tamanho do dataset final
downsampled_dataset = lm_datasets["train"].train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)
print(downsampled_dataset)

# Login no huggingface para salvar o modelo
#from huggingface_hub import notebook_login

#notebook_login()

# Prepara os TrainingArguments
from transformers import TrainingArguments

batch_size = 32
# Show the training loss with every epoch
logging_steps = len(downsampled_dataset["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]

training_args = TrainingArguments(
    output_dir="BERTweetBRLogger2",
    logging_dir = "BERTweetBRLogger2_logs",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    fp16=True,
    logging_steps=logging_steps,
)

# Preparar o Trainer
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=downsampled_dataset["train"],
    eval_dataset=downsampled_dataset["test"],
    data_collator=data_collator,
)

# Testes de perplexidade
import math

eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

train_result = trainer.train()

eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

# Testando log de resultados
metrics = train_result.metrics
max_train_samples = len(downsampled_dataset["train"])
metrics["train_samples"] = min(max_train_samples, len(downsampled_dataset["train"]))

# save train results
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)

trainer.save_model("BERTweetBRLogger2")

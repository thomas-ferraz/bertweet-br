# Valores para configuracao
model_checkpoint = 'neuralmind/bert-base-portuguese-cased'
tokenizer_checkpoint = 'neuralmind/bert-base-portuguese-cased'
chunk_size = 128
batch_size = 32
train_size = 20000
test_size = int(0.1 * train_size)
learning_rate = 2e-5
weight_decay = 0.01
output_dir = "BERTweetBR2" # Nao use caracteres especiais, nem . ou /
logging_dir = "BERTweetBR2_logs" # Nao use caracteres especiais, nem . ou /
evaluation_strategy="steps"
overwrite_output_dir=True
fp16=False



# Funcao para tokenizacao
def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

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
    
    
    
# Pega o model
from transformers import AutoModelForMaskedLM
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

# Pega o tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)

# Pega o Data Collator
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)



# Prepara datasets
from datasets import load_dataset
raw_dataset = load_dataset('text', data_files={'train': ['./tweets/text/text_1.txt','./tweets/text/text_2.txt','./tweets/text/text_3.txt','./tweets/text/text_4.txt','./tweets/text/text_5.txt','./tweets/text/text_6.txt','./tweets/text/text_7.txt','./tweets/text/text_8.txt','./tweets/text/text_9.txt','./tweets/text/text_10.txt','./tweets/text/text_11.txt','./tweets/text/text_12.txt','./tweets/text/text_13.txt','./tweets/text/text_14.txt','./tweets/text/text_15.txt','./tweets/text/text_16.txt','./tweets/text/text_17.txt','./tweets/text/text_18.txt','./tweets/text/text_19.txt','./tweets/text/text_20.txt','./tweets/text/text_21.txt','./tweets/text/text_22.txt','./tweets/text/text_23.txt','./tweets/text/text_24.txt','./tweets/text/text_25.txt']})
print(raw_dataset)

# Diminuir tamanho do dataset
downsampled_dataset = raw_dataset["train"].train_test_split(train_size=train_size,test_size=test_size, seed=42)
print(downsampled_dataset)

# Tokenizando datasets
tokenized_datasets = downsampled_dataset.map(
    tokenize_function, batched=True, remove_columns=["text"]
)
print(tokenized_datasets)

# Aplicando group_texts para dataset tokenizado
final_dataset = tokenized_datasets.map(group_texts, batched=True)
print(final_dataset)



# Carrega metrica de perplexidade
from datasets import load_metric
metric = load_metric("perplexity")



# Muda verbosidade do transformers
import transformers
transformers.logging.set_verbosity_info()



# Mostra log a cada step definido abaixo
logging_steps = len(final_dataset["train"]) // batch_size



# Prepara os TrainingArguments
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir=output_dir,
    logging_dir = logging_dir,
    overwrite_output_dir=overwrite_output_dir,
    evaluation_strategy=evaluation_strategy,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    fp16=fp16,
    logging_steps=logging_steps,
)



# Prepara o Trainer
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=final_dataset["train"],
    eval_dataset=final_dataset["test"],
    data_collator=data_collator,
)



# Coleta perplexidade antes de treinar, somente avaliando
import math
eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
initial_perplexity = math.exp(eval_results['eval_loss'])



# Treina
train_result = trainer.train('./BERTweetBR2/checkpoint-11500')



# Coleta perplexidade apos treinar, avaliando
eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")



# Coletando metricas do resultado de train()
metrics = train_result.metrics
metrics["train_samples"] = len(final_dataset["train"])

# Save train results
trainer.log_metrics("all", metrics)
trainer.save_metrics("all", metrics)



# Cria log do historico do obj do Trainer
with open(str(logging_dir)+'/trainer_logs.txt', 'w') as f:
	for obj in trainer.state.log_history:
		f.write(str(obj))
		f.write('\n')
	f.write('\n\n\n')
	f.write(str(metrics))
	f.write('\n\n\n')
	f.write('Initial Perplexity = '+str(initial_perplexity))
	f.write('\n')
	f.write('Final Perplexity = '+str(math.exp(eval_results['eval_loss'])))



# Salva modelo treinado
trainer.save_model(output_dir)

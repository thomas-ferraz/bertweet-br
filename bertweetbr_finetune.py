# Valores para configuracao
model_checkpoint = 'neuralmind/bert-base-portuguese-cased'
tokenizer_checkpoint = 'neuralmind/bert-base-portuguese-cased'
chunk_size = 282
batch_size = 32
train_size = 100
test_size = int(0.1 * train_size)
learning_rate = 2e-5
weight_decay = 0.01
output_dir = "BERTweetBR2_sentiment" # Nao use caracteres especiais, nem . ou /
logging_dir = "BERTweetBR2_sentiment_logs" # Nao use caracteres especiais, nem . ou /
evaluation_strategy="steps"
overwrite_output_dir=True
fp16=False



# Funcao para tokenizacao
def tokenize_function(examples):
    examples["labels"] = examples["sentiment"]
    result = tokenizer(examples["tweet_text"], truncation=True)
    #if tokenizer.is_fast:
        #result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result



print('\nETAPA - COLETA DE MODELO E TOKENIZADOR\n')    
# Pega o model
from transformers import AutoModelForPreTraining, AutoModelForTokenClassification, AutoModelForSequenceClassification, BertForPreTraining, BertModel, AutoModel
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=3)

# Pega o tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)

# Pega o Data Collator
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)



print('\nETAPA - COLETA DATASET RAW\n')
# Prepara datasets
# Negative label = 0
# Positive label = 1
# Neutral label = 2
from datasets import load_dataset, ClassLabel
raw_datasets = load_dataset('csv', delimiter=';', data_files={'train': ['./kaggle/trainingdatasets/Train3Classes.csv'], 'validation':['./kaggle/testdatasets/Test3classes.csv'], 'test': ['./kaggle/testdatasets/Test3classes.csv']})

print('\nETAPA - MUDA COLUNA SENTIMENT DE INT PARA CLASSLABEL\n')
raw_datasets = raw_datasets.class_encode_column("sentiment")
#feat_sentiment = ClassLabel(num_classes=3, names = ['0', '1', '2'], names_file=None)
#raw_datasets = raw_datasets.cast_column("sentiment", feat_sentiment)

print('\nETAPA - FEATURES DE RAW_DATASET\n')
raw_train_dataset = raw_datasets["train"]
print(raw_train_dataset.features)

print('\nETAPA - DOWNSAMPLE DO DATASET\n')


print('\nETAPA - TOKENIZA DATASET\n')
# Tokenizando datasets
tokenized_datasets = raw_datasets.map(
    tokenize_function, batched=True, remove_columns=["id", "tweet_date", "query_used"]
)
tokenized_datasets = tokenized_datasets.class_encode_column("labels")

print('\nETAPA - FEATURES DE TOKENIZED_DATASET\n')
tokenized_train_dataset = tokenized_datasets["train"]
print(tokenized_train_dataset.features)



# Teste aplicando data collator
samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["tweet_text"]}
print([len(x) for x in samples["input_ids"]])

batch = data_collator(samples)
print({k: v.shape for k, v in batch.items()})



# Muda verbosidade do transformers
import transformers
transformers.logging.set_verbosity_info()

# Mostra log a cada step definido abaixo
logging_steps = len(tokenized_datasets["train"]) // batch_size



print('\nETAPA - TREINO\n')
from transformers import TrainingArguments

training_args = TrainingArguments("test-trainer")

from transformers import Trainer

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print(trainer.train_dataset[0])
print(tokenizer.decode(trainer.train_dataset[0]["input_ids"]))
print(trainer.train_dataset[0].keys())
print(trainer.train_dataset[0]["attention_mask"])
print(len(trainer.train_dataset[0]["attention_mask"]) == len(trainer.train_dataset[0]["input_ids"]))
print(trainer.train_dataset[0]["labels"])
print(trainer.train_dataset.features["labels"].names)

for batch in trainer.get_train_dataloader():
    print(batch)
    break

trainer.train()

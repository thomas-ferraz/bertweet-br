# Valores para configuracao
model_checkpoint = './BERTweetBR3_sentiment'
tokenizer_checkpoint = 'neuralmind/bert-base-portuguese-cased'
chunk_size = 282
batch_size = 8
train_size = 40000
test_size = int(0.1 * train_size)
learning_rate = 2e-5
weight_decay = 0.01
output_dir = "BERTweetBR2_sentiment" # Nao use caracteres especiais, nem . ou /
logging_dir = "BERTweetBR2_sentiment_logs" # Nao use caracteres especiais, nem . ou /
evaluation_strategy="epoch"
overwrite_output_dir=True
fp16=False



import torch
print(torch.cuda.is_available())

print('\n ETAPA - DEFINICAO DE MODELO E TOKENIZADOR \n')    
# Pega o model
from transformers import AutoModelForPreTraining, AutoModelForTokenClassification, AutoModelForSequenceClassification, BertForPreTraining, BertModel, AutoModel
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=3)

# Pega o tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)

# Pega o Data Collator
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)



from transformers import pipeline

classifier = pipeline('sentiment-analysis', model=model_checkpoint, tokenizer=tokenizer_checkpoint)

test_strings = ['Nossa! Muito obrigada :)', 'Um teste horrivel sem emoji', 'Trist dia familia', 'Apesar de achar que as eleicoes foram horriveis, minha mae esta muito feliz hoje', 'Mas que mandinga danada seus abofe doido', 'Olha, nem foi tao ruim quanto pensava', 'Foi otimo']

results = classifier(test_strings)

for result in results:
	print(result)
	
tokens = tokenizer.tokenize('Um teste horrivel sem emoji')
token_ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = tokenizer('Um teste horrivel sem emoji')

print(tokens)
print(token_ids)
print(input_ids)

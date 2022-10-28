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
import pandas as pd

classifier = pipeline('sentiment-analysis', model=model_checkpoint, tokenizer=tokenizer_checkpoint)

original_csv = pd.read_csv('./kaggle/testdatasets/Test3classesClean.csv', sep=";")#['Nossa! Muito obrigada :)', 'Um teste horrivel sem emoji', 'Trist dia familia', 'Apesar de achar que as eleicoes foram horriveis, minha mae esta muito feliz hoje', 'Mas que mandinga danada seus abofe doido', 'Olha, nem foi tao ruim quanto pensava', 'Foi otimo', 'Mas que dia hein?', 'Po, tentei e tentei e não consegui', 'Estou tendo problemas com seu produto']

original_csv = original_csv.head(500)

test_strings = original_csv['tweet_text'].tolist()

y_true = original_csv['sentiment'].tolist()

print(y_true)

results = classifier(test_strings)

print(results)

y_pred = []

for result in results:
	if result['label'] == "LABEL_0":
		y_pred.append(0)
	elif result['label'] == "LABEL_1":
		y_pred.append(1)
	else:
		y_pred.append(2)
print(y_pred)	

from sklearn import metrics
print(metrics.precision_score(y_true, y_pred, average='macro'))
print(metrics.recall_score(y_true, y_pred, average='macro'))
print(metrics.f1_score(y_true, y_pred, average='macro'))
#tokens = tokenizer.tokenize('Um teste horrivel sem emoji')
#token_ids = tokenizer.convert_tokens_to_ids(tokens)
#input_ids = tokenizer('Um teste horrivel sem emoji')

#print(tokens)
#print(token_ids)
#print(input_ids)

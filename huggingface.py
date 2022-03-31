from datasets import load_dataset

#dataset = load_dataset('oscar', 'unshuffled_deduplicated_it')

from tqdm.auto import tqdm

#text_data = []
#file_count = 0

#for sample in tqdm(dataset['train']):
#    sample = sample['text'].replace('\n', '')
#    text_data.append(sample)
#    if len(text_data) == 10_000:
#        # once we git the 10K mark, save to file
#        with open(f'./data/text/oscar_it/text_{file_count}.txt', 'w', encoding='utf-8') as fp:
#            fp.write('\n'.join(text_data))
#        text_data = []
#        file_count += 1
# after saving in 10K chunks, we will have ~2082 leftover samples, we save those now too
#with open(f'./data/text/oscar_it/text_{file_count}.txt', 'w', encoding='utf-8') as fp:
#    fp.write('\n'.join(text_data))
    
from pathlib import Path
paths = [str(x) for x in Path('./data/text/oscar_it').glob('**/*.txt')]

from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(files=paths[:5], vocab_size=30_522, min_frequency=2,
                special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'])

import os

os.mkdir('./filiberto')

tokenizer.save_model('filiberto')

from transformers import RobertaTokenizer

# initialize the tokenizer using the tokenizer we initialized and saved to file
tokenizer = RobertaTokenizer.from_pretrained('filiberto', max_len=512)

# test our tokenizer on a simple sentence
tokens = tokenizer('ciao, come va?')
print(tokens)

# input pipeline
with open('./data/text/oscar_it/text_0.txt', 'r', encoding='utf-8') as fp:
	lines = fp.read().split('\n')
	
batch = tokenizer(lines, max_length=512, padding='max_length', truncation=True)
len(batch)
   
import torch

labels = torch.tensor([x.ids for x in batch])
mask = torch.tensor([x.attention_mask for x in batch])

# make copy of labels tensor, this will be input_ids
input_ids = labels.detach().clone()
# create random array of floats with equal dims to input_ids
rand = torch.rand(input_ids.shape)
# mask random 15% where token is not 0 [PAD], 1 [CLS], or 2 [SEP]
mask_arr = (rand < .15) * (input_ids != 0) * (input_ids != 1) * (input_ids != 2)
# loop through each row in input_ids tensor (cannot do in parallel)
for i in range(input_ids.shape[0]):
    # get indices of mask positions from mask array
    selection = torch.flatten(mask_arr[i].nonzero()).tolist()
    # mask input_ids
    input_ids[i, selection] = 3  # our custom [MASK] token == 3
    
encodings = {'input_ids': input_ids, 'attention_mask': mask, 'labels': labels}

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        # store encodings internally
        self.encodings = encodings

    def __len__(self):
        # return the number of samples
        return self.encodings['input_ids'].shape[0]

    def __getitem__(self, i):
        # return dictionary of input_ids, attention_mask, and labels for index i
        return {key: tensor[i] for key, tensor in self.encodings.items()}

dataset = Dataset(encodings)

loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)



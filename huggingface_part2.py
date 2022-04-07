from pathlib import Path
from transformers import RobertaTokenizer
import torch
from pathlib import Path
from tqdm.auto import tqdm
import random
import os



input_ids = torch.load('./filiberto_training/input_ids.pt')
mask = torch.load('./filiberto_training/attention_mask.pt')
labels = torch.load('./filiberto_training/labels.pt')

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

from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size=30_522,  # we align this to the tokenizer vocab set in previous notebook
    max_position_embeddings=514,
    hidden_size=768,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1
)

from transformers import RobertaForMaskedLM

model = RobertaForMaskedLM(config)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# and move our model over to the selected device
model.to(device)

from transformers import AdamW

# activate training mode
model.train()
# initialize optimizer
optim = AdamW(model.parameters(), lr=1e-5)

# optionally, if using tensorboard, initialize writer object
#writer = torch.utils.tensorboard.SummaryWriter()

from tqdm import tqdm  # for our progress bar

epochs = 1  # trained for 4 in total
step = 0

for epoch in range(epochs):
    # setup loop with TQDM and dataloader
    loop = tqdm(loader, leave=True)
    for batch in loop:
        # initialize calculated gradients (from prev step)
        optim.zero_grad()
        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        # process
        outputs = model(input_ids, attention_mask=attention_mask,
                        labels=labels)
        # extract loss
        loss = outputs.loss
        # take loss for tensorboard
        #writer.add_scalar('Loss/train', loss, step)
        # calculate loss for every parameter that needs grad update
        loss.backward()
        # update parameters
        optim.step()
        # print relevant info to progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())
        step += 1
        

model.save_pretrained('./filiberto')

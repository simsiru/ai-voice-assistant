import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertModel


def get_dataframe(data_path):
    '''
    intents = {
        intent1: 'lights',
        intent2: 'weather',
        intent3: 'surveillance'
    }
    '''

    with open(data_path) as f:
        data = f.readlines()

    print(data)

    intents = []
    utterances = []

    for utterance in data[:59]:
        intents.append(utterance[6])
        utterances.append(utterance[8:])

    #len(intents), len(utterances)


    data_dict = {
        'intents': intents,
        'utterances': utterances
    }

    df = pd.DataFrame(data_dict)

    print(df.head())

    print(df.isnull().sum())

    mapping = {
        '1': 0,
        '2': 1#,
        #'3': 2
    }

    df['intents'] = df['intents'].map(mapping)

    print(df.head())
    print(df['intents'].value_counts())

    return df


def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)


class IntentDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
        self.maxlen = 256
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        utterance = self.df['utterances'].iloc[index].split()
        utterance = ' '.join(utterance)
        intent = int(self.df['intents'].iloc[index])

        encodings = self.tokenizer.encode_plus(
            utterance,
            add_special_tokens=True,
            max_length=self.maxlen,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encodings.input_ids.flatten(),
            'attention_mask': encodings.attention_mask.flatten(),
            'labels': torch.tensor(intent, dtype=torch.long)
        }



class IntentClassifier(nn.Module):
    def __init__(self):
        super(IntentClassifier, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.drop0 = nn.Dropout(0.25)
        self.linear1 = nn.Linear(3072, 512)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.25)
        self.linear2 = nn.Linear(512, 3)
        self.relu2 = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids, attention_mask)
        last_hidden_state = outputs[0]
        pooled_output = torch.cat(tuple([last_hidden_state[:, i] for i in [-4, -3, -2, -1]]), dim=-1)
        x = self.drop0(pooled_output)
        x = self.relu1(self.linear1(x))
        x = self.drop1(x)
        x = self.relu2(self.linear2(x))
        return x



def train_intent_classifier():

    df = get_dataframe("intent_classifier_data/intent_dataset.txt")


    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_dataset = IntentDataset(train_df)
    valid_dataset = IntentDataset(test_df)


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=8
    )


    for batch in train_loader:
        print(batch['input_ids'].shape)
        print(batch['attention_mask'].shape)
        print(batch['labels'].shape)
        break




    model = IntentClassifier()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)


    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    epochs = 5


    for epoch in range(epochs):
        # TRAIN
        model.train()
        train_loop = tqdm(train_loader)
        for batch in train_loop:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            output = model(input_ids, attention_mask)
            loss = criterion(output, labels)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loop.set_description(f"Training Epoch: {epoch}")
            train_loop.set_postfix(loss=loss.item())

        # VALIDATION
        model.eval()
        valid_loop = tqdm(valid_loader)
        for batch in valid_loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            output = model(input_ids, attention_mask)
            loss = criterion(output, labels)

            valid_loop.set_description(f"Validation Epoch: {epoch}")
            valid_loop.set_postfix(loss=loss.item())



    torch.save(model.state_dict(), 'intent_classifier_data/intent_classifier.pth')



    #test_sample = test_df['utterances'].iloc[100]
    #original_label = test_df['intents'].iloc[100]

    #print(test_sample)
    #print(original_label)

    data_dict = {
        'intents': [1],
        'utterances': ["is there snow today"]#["turn off the lights"]["is there rain outside"]
    }

    test_sample = pd.DataFrame(data_dict)
    print(test_sample.head())

    test_sample = test_sample['utterances'].iloc[0].split()
    test_sample = ' '.join(test_sample)
    #print(test_sample)

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    encodings = tokenizer.encode_plus(
        test_sample,
        add_special_tokens=True,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    with torch.no_grad():
        model.to('cpu')
        preds = model(encodings['input_ids'].to('cpu'), encodings['attention_mask'].to('cpu'))
        preds = np.argmax(preds)
        output = preds.item()
        print(output+1)


if __name__ == "__main__":
    train_intent_classifier()
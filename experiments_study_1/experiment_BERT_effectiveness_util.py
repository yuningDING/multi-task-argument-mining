import gc
import os
import random

import numpy as np
import torch
from sklearn.metrics import (accuracy_score,
                             f1_score,
                             recall_score,
                             confusion_matrix,
                             cohen_kappa_score)
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, AutoConfig, AutoModel


class Constants:
    TARGET_LIST = ["Ineffective", "Adequate", "Effective"]


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class FeedbackPrizeModel(torch.nn.Module):
    def __init__(self):
        super(FeedbackPrizeModel, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, len(Constants.TARGET_LIST))

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert_model(input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids)
        output = self.dropout(output.pooler_output)
        output = self.linear(output)

        return output


class FeedbackPrizeWithCustomFeatureModel(torch.nn.Module):
    def __init__(self, model_name, num_extra_dims, num_labels):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config)
        num_hidden_size = self.transformer.config.hidden_size
        self.classifier = torch.nn.Linear(num_hidden_size + num_extra_dims, num_labels)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, len(Constants.TARGET_LIST))

    def forward(self, input_ids, extra_data, attention_mask=None):
        hidden_states = self.transformer(input_ids=input_ids,
                                         attention_mask=attention_mask)  # [batch size, sequence length, hidden size]
        cls_embeds = hidden_states.last_hidden_state[:, 0, :]  # [batch size, hidden size]
        concat = torch.cat((cls_embeds, extra_data), dim=-1)  # [batch size, hidden size+num extra dims]
        output = self.classifier(concat)  # [batch size, num labels]
        return output


def build_model_tokenizer(with_custom_feature, num_extra_dims, model_path=None):
    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Modell
    if with_custom_feature:
        model = FeedbackPrizeWithCustomFeatureModel('bert-base-uncased', num_extra_dims, 3)
    else:
        model = FeedbackPrizeModel()

    if model_path is not None:
        model.load_state_dict(torch.load(model_path))

    return tokenizer, model


class FeedbackPrizeDataset(Dataset):
    def __init__(self, data, max_len, tokenizer, data_path, plus_text, extra_feature):
        self.data = data
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.discourse_text = data['discourse_text'].values
        self.discourse_type = data['discourse_type'].values
        self.prompt = data['prompt'].values
        self.targets = data[Constants.TARGET_LIST].values
        self.essay_id = data['essay_id'].values
        self.plus_text = plus_text
        self.extra_feature = extra_feature

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        discourse_text = self.discourse_text[index]
        inputs = self.tokenizer.encode_plus(discourse_text.lower(),
                                            truncation=True,
                                            padding='max_length',
                                            add_special_tokens=True,
                                            return_attention_mask=True,
                                            return_token_type_ids=True,
                                            max_length=self.max_len,
                                            return_tensors='pt')
        if self.plus_text:
            essay_path = os.path.join(self.data_path, f"{self.essay_id[index]}.txt")
            # --- discourse [SEP] essay ---
            essay = open(essay_path, 'r').read()
            inputs = self.tokenizer(discourse_text.lower(), essay.lower(), truncation=True, padding='max_length',
                                    add_special_tokens=True, return_attention_mask=True, return_token_type_ids=True,
                                    max_length=self.max_len, return_tensors='pt')

        input_ids = inputs['input_ids'].flatten()
        attention_mask = inputs['attention_mask'].flatten()
        token_type_ids = inputs['token_type_ids'].flatten()
        targets = torch.FloatTensor(self.targets[index])

        if len(self.extra_feature) > 0:
            return {'input_ids': input_ids, 'attention_mask': attention_mask,
                    'extra_data': torch.FloatTensor(self.extra_feature[index]), 'token_type_ids': token_type_ids,
                    'targets': targets}
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids,
                'targets': targets}


# Loss Function
def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


def get_optimizer(model, learning_rate):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    return optimizer


val_targets = []
val_outputs = []


def train_model(n_epochs,
                train_loader,
                val_loader,
                test_loader,
                model, lr,
                device, extra_data=None):
    optimizer = get_optimizer(model, lr)
    model.to(device)
    for epoch in range(n_epochs):
        train_loss = 0
        val_loss = 0
        model.train()
        print(f' Epoch: {epoch + 1} - Train Set '.center(50, '='))
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            input_ids = batch['input_ids'].to(device, dtype=torch.long)
            attention_mask = batch['attention_mask'].to(device, dtype=torch.long)
            token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)
            targets = batch['targets'].to(device, dtype=torch.float)
            if extra_data is None:
                outputs = model(input_ids, attention_mask, token_type_ids)
            else:
                extra_data = batch['extra_data'].to(device, dtype=torch.long)
                outputs = model(input_ids, extra_data, attention_mask)
            optimizer.zero_grad()
            loss = loss_fn(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))
            del input_ids, attention_mask, token_type_ids, targets, outputs
            gc.collect()

        print(f' Epoch: {epoch + 1} - Validation Set '.center(50, '='))
        model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(val_loader)):
                input_ids = data['input_ids'].to(device, dtype=torch.long)
                attention_mask = data['attention_mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.float)
                if extra_data is None:
                    outputs = model(input_ids, attention_mask, token_type_ids)
                else:
                    extra_data = data['extra_data'].to(device, dtype=torch.long)
                    outputs = model(input_ids, extra_data, attention_mask)
                loss = loss_fn(outputs, targets)
                val_loss = val_loss + ((1 / (batch_idx + 1)) * (loss.item() - val_loss))
                val_targets.extend(targets.cpu().detach().numpy().tolist())
                val_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
                del input_ids, attention_mask, token_type_ids, targets, outputs
                gc.collect()
            train_loss = train_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)
            print('Epoch: {} \tAvgerage Training Loss: {:.6f} \tAverage Validation Loss: {:.6f} \n'.format(
                epoch + 1,
                train_loss,
                val_loss
            ))

    print('Test')
    model.eval()
    test_targets = []
    test_outputs = []
    test_loss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader)):
            input_ids = data['input_ids'].to(device, dtype=torch.long)
            attention_mask = data['attention_mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            if extra_data is None:
                outputs = model(input_ids, attention_mask, token_type_ids)
            else:
                extra_data = data['extra_data'].to(device, dtype=torch.long)
                outputs = model(input_ids, extra_data, attention_mask)
            loss = loss_fn(outputs, targets)
            test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.item() - test_loss))
            test_targets.extend(targets.cpu().detach().numpy().tolist())
            test_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    test_outputs_labels = np.array([np.argmax(a) for a in test_outputs])
    test_targets_labels = np.array([np.argmax(a) for a in test_targets])
    accuracy = accuracy_score(test_targets_labels, test_outputs_labels)
    recall_micro = recall_score(test_targets_labels, test_outputs_labels, average='micro')
    recall_macro = recall_score(test_targets_labels, test_outputs_labels, average='macro')
    f1_score_micro = f1_score(test_targets_labels, test_outputs_labels, average='micro')
    f1_score_macro = f1_score(test_targets_labels, test_outputs_labels, average='macro')
    qwk = cohen_kappa_score(test_targets_labels, test_outputs_labels, weights='quadratic')
    print(f"Test Loss: {round(test_loss, 4)}")
    print(f"Accuracy Score: {round(accuracy, 4)}")
    print(f"Recall (Micro): {round(recall_micro, 4)}")
    print(f"Recall (Macro): {round(recall_macro, 4)}")
    print(f"F1 Score (Micro): {round(f1_score_micro, 4)}")
    print(f"F1 Score (Macro): {round(f1_score_macro, 4)} \n")
    print(f"QWK: {round(qwk, 4)}")
    cm = confusion_matrix(test_targets_labels, test_outputs_labels)
    print("Confusion Matrix:")
    print(cm)

    return model


def model_predict(device, model, test_loader, extra_data=None):
    print('Test')
    model.eval()
    test_targets = []
    test_outputs = []
    test_loss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader)):
            input_ids = data['input_ids'].to(device, dtype=torch.long)
            attention_mask = data['attention_mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            if extra_data is None:
                outputs = model(input_ids, attention_mask, token_type_ids)
            else:
                extra_data = data['extra_data'].to(device, dtype=torch.long)
                outputs = model(input_ids, extra_data, attention_mask, token_type_ids)
            loss = loss_fn(outputs, targets)
            test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.item() - test_loss))
            test_targets.extend(targets.cpu().detach().numpy().tolist())
            test_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    test_outputs_labels = np.array([np.argmax(a) for a in test_outputs])
    test_targets_labels = np.array([np.argmax(a) for a in test_targets])
    accuracy = accuracy_score(test_targets_labels, test_outputs_labels)
    recall_micro = recall_score(test_targets_labels, test_outputs_labels, average='micro')
    recall_macro = recall_score(test_targets_labels, test_outputs_labels, average='macro')
    f1_score_micro = f1_score(test_targets_labels, test_outputs_labels, average='micro')
    f1_score_macro = f1_score(test_targets_labels, test_outputs_labels, average='macro')
    qwk = cohen_kappa_score(test_targets_labels, test_outputs_labels, weights='quadratic')
    print(f"Test Loss: {round(test_loss, 4)}")
    print(f"Accuracy Score: {round(accuracy, 4)}")
    print(f"Recall (Micro): {round(recall_micro, 4)}")
    print(f"Recall (Macro): {round(recall_macro, 4)}")
    print(f"F1 Score (Micro): {round(f1_score_micro, 4)}")
    print(f"F1 Score (Macro): {round(f1_score_macro, 4)} \n")
    print(f"QWK: {round(qwk, 4)}")
    cm = confusion_matrix(test_targets_labels, test_outputs_labels)
    print("Confusion Matrix:")
    print(cm)


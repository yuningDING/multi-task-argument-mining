import os
import sys
import random
from typing import Optional, Tuple, Union

from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import LongformerTokenizerFast, AutoTokenizer, LongformerForTokenClassification, LongformerModel, \
    AutoModel, LongformerPreTrainedModel
from transformers.models.longformer.modeling_longformer import LongformerTokenClassifierOutput

pd.options.mode.chained_assignment = None


class Constants:
    ARGU_OUTPUT_LABELS = ['O-Argument', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim',
                          'B-Counterclaim',
                          'I-Counterclaim', 'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence',
                          'B-Concluding Statement',
                          'I-Concluding Statement']
    ARGU_LABELS_TO_IDS = {v: k for k, v in enumerate(ARGU_OUTPUT_LABELS)}
    ARGU_IDS_TO_LABELS = {k: v for k, v in enumerate(ARGU_OUTPUT_LABELS)}

    EFFECT_OUTPUT_LABELS = ['O-Effectivness', 'B-Effective', 'I-Effective', 'B-Adequate', 'I-Adequate', 'B-Ineffective',
                            'I-Ineffective']
    # EFFECT_LABELS_TO_IDS = {'O-Effectivness':15, 'B-Effective':16, 'I-Effective':17, 'B-Adequate':18, 'I-Adequate':19, 'B-Ineffective':20, 'I-Ineffective':21}
    # EFFECT_IDS_TO_LABELS = {15:'O-Effectivness', 16:'B-Effective', 17:'I-Effective', 18:'B-Adequate', 19:'I-Adequate', 20:'B-Ineffective', 21:'I-Ineffective'}
    EFFECT_LABELS_TO_IDS = {v: k for k, v in enumerate(EFFECT_OUTPUT_LABELS)}
    EFFECT_IDS_TO_LABELS = {k: v for k, v in enumerate(EFFECT_OUTPUT_LABELS)}

    replace_chars = {"Ë": "E", "´": "'", "\x94": "", "\x93": "", "¨": "'", "å": "a", "\x91": "", "\x97": ""}


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class LongformerForMultiTaskTokenClassification(LongformerPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, num_labels_taskA, num_labels_taskB):
        super().__init__(config)
        self.num_labels_taskA = num_labels_taskA
        self.num_labels_taskB = num_labels_taskB

        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.dropout_taskA = torch.nn.Dropout(config.hidden_dropout_prob)
        self.dropout_taskB = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier_taskA = torch.nn.Linear(config.hidden_size, num_labels_taskA)
        self.classifier_taskB = torch.nn.Linear(config.hidden_size, num_labels_taskB)

        # Initialize weights and apply final processing
        self.post_init()

    # @add_start_docstrings_to_model_forward(LONGFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     processor_class=_TOKENIZER_FOR_DOC,
    #     checkpoint="brad1141/Longformer-finetuned-norm",
    #     output_type=LongformerTokenClassifierOutput,
    #     config_class=_CONFIG_FOR_DOC,
    #     expected_output=(
    #         "['Evidence', 'Evidence', 'Evidence', 'Evidence', 'Evidence', 'Evidence', 'Evidence', 'Evidence',"
    #         " 'Evidence', 'Evidence', 'Evidence', 'Evidence']"
    #     ),
    #     expected_loss=0.63,
    # )
    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            global_attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            taskA_labels: Optional[torch.Tensor] = None,
            taskB_labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, LongformerTokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            head_mask=head_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output_taskA = self.dropout_taskA(sequence_output)
        logits_taskA = self.classifier_taskA(sequence_output)

        sequence_output_taskB = self.dropout_taskB(sequence_output)
        logits_taskB = self.classifier_taskB(sequence_output)

        loss_taskA = None
        loss_taskB = None
        if (taskA_labels is not None) and (taskB_labels is not None):
            loss_fct = CrossEntropyLoss()

            loss_taskA = loss_fct(logits_taskA.view(-1, self.num_labels_taskA), taskA_labels.view(-1))
            loss_taskB = loss_fct(logits_taskB.view(-1, self.num_labels_taskB), taskB_labels.view(-1))

        if not return_dict:
            output_taskA = (logits_taskA,) + outputs[2:]
            output_taskB = (logits_taskB,) + outputs[2:]

            return (((loss_taskA + loss_taskB) / 2,) + [output_taskA, output_taskB]) if (
                        (loss_taskA is not None) and (loss_taskB is not None)) else [output_taskA, output_taskB]

        loss = None
        if loss_taskA != None and loss_taskB != None:
            # loss = (loss_taskA + loss_taskB) / 2,
            loss = torch.add(loss_taskA, loss_taskB)
            loss = torch.div(loss, 2)
        else:
            loss = None

        return LongformerTokenClassifierOutput(
            #loss=(loss_taskA + loss_taskB) /
            loss=loss,
            logits=[logits_taskA, logits_taskB],
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            global_attentions=outputs.global_attentions,
        )


def agg_essays(folder):
    names, texts = [], []
    for f in tqdm(list(os.listdir(folder))):
        names.append(f.replace('.txt', ''))
        next_text = open(folder + '/' + f, 'r', encoding='utf-8').read()
        for error_char in Constants.replace_chars.keys():
            next_text = next_text.replace(error_char, Constants.replace_chars[error_char])
        texts.append(next_text)
    df_texts = pd.DataFrame({'id': names, 'text': texts})

    df_texts['text_split'] = df_texts.text.str.split()
    print('Completed tokenizing texts.')
    return df_texts


def ner(df_texts, df_train):
    all_discourse = []
    all_effectiveness = []
    for _, row in tqdm(df_texts.iterrows(), total=len(df_texts)):
        # print(row['id'])
        total = len(row.text_split)
        # print("essay words: ", row.text_split)
        # print("essay length: ", total)
        discourse = ['O-Argument'] * total
        effectiveness = ['O-Effectivness'] * total

        for _, row2 in df_train.loc[df_train['essay_id'] == row['id']].iterrows():
            # print(row2)
            discourse_gold = row2['discourse_type']
            effectiveness_gold = row2['discourse_effectiveness']
            list_ix = [int(x) for x in row2['prediction_string'].split(' ')]

            # To allow setting either task entirely to O
            if discourse_gold != 'O-Argument':
                discourse[list_ix[0]] = f'B-{discourse_gold}'
                for k in list_ix[1:]:
                    try:
                        discourse[k] = f'I-{discourse_gold}'
                        # effectiveness[k] = f'I-{effectiveness_gold}'
                    except IndexError:
                        print(row['id'])
                        print(row2['discourse_text'])
                        print('predictionstring index:', k)
                        print('max length of text:', total)

            if effectiveness_gold != 'O-Effectivness':
                effectiveness[list_ix[0]] = f'B-{effectiveness_gold}'
                for k in list_ix[1:]:
                    try:
                        # discourse[k] = f'I-{discourse_gold}'
                        effectiveness[k] = f'I-{effectiveness_gold}'
                    except IndexError:
                        print(row['id'])
                        print(row2['discourse_text'])
                        print('predictionstring index:', k)
                        print('max length of text:', total)

        all_discourse.append(discourse)
        all_effectiveness.append(effectiveness)

    df_texts['discourse_BIO'] = all_discourse
    #print(df_texts['discourse_BIO'])
    df_texts['effectiveness_BIO'] = all_effectiveness
    #print(df_texts['effectiveness_BIO'])

    print('Completed mapping discourse to each token.')
    return df_texts


def preprocess(text_folder, df_gold):
    df_texts = agg_essays(text_folder)
    # print('before mergen')
    # print(df_texts)
    df_texts =df_texts[df_texts['id'].isin(set(df_gold['essay_id'].unique()))].reset_index()
    # df_texts = df_texts.merge(df_gold, left_on='id', right_on='essay_id', how='inner')
    # print('after mergen')
    # print(df_texts)
    df_texts = ner(df_texts, df_gold)
    return df_texts


def build_model_tokenizer_multi_task(model_name, num_labels_A, num_labels_B, model_path=None):
    # Tokenizer
    tokenizer = LongformerTokenizerFast.from_pretrained(model_name, add_prefix_space=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Modell
    model = LongformerForMultiTaskTokenClassification.from_pretrained(model_name, num_labels_A, num_labels_B)

    # ADD PAD TOKEN TO MODEL
    model.resize_token_embeddings(len(tokenizer))
    model.longformer.embeddings.word_embeddings.padding_idx = 1
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    return model, tokenizer


def build_model_tokenizer(model, model_path=None):
    # Tokenizer
    if 'longformer' in model:
        tokenizer = LongformerTokenizerFast.from_pretrained(model, add_prefix_space=True)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    else:
        tokenizer = AutoTokenizer.from_pretrained(model, add_prefix_space=True)

    # Modell
    if 'longformer' in model:
        model = LongformerForTokenClassification.from_pretrained(model, num_labels=15)
        # ADD PAD TOKEN TO MODEL
        model.resize_token_embeddings(len(tokenizer))
        model.longformer.embeddings.word_embeddings.padding_idx = 1
        if model_path is not None:
            model.load_state_dict(torch.load(model_path))
    else:
        model = AutoModel.from_pretrained(model, num_labels=15)
        if model_path is not None:
            model.load_state_dict(torch.load(model_path))

    return model, tokenizer


class FeedbackPrizeDataset_MultiTask(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        # step 1: get the sentence and word labels
        # print("--- TEXT SPLIT ---")
        # print(len(self.data.text_split))
        # print(self.data.text_split)
        # print("---")
        sentence = self.data.text_split[index]
        argu_labels = self.data.discourse_BIO[index]
        effective_labels = self.data.effectiveness_BIO[index]

        # step 2: use tokenizer to encode sentence (includes padding/truncation up to max length)
        encoding = self.tokenizer(sentence,
                                  is_split_into_words=True,
                                  padding='max_length',
                                  return_offsets_mapping=True,
                                  truncation=True,
                                  max_length=self.max_len)

        # step 3: create token labels only for first word pieces of each tokenized word
        argument_labels = [Constants.ARGU_LABELS_TO_IDS[label] for label in argu_labels]
        effectiveness_labels = [Constants.EFFECT_LABELS_TO_IDS[label] for label in effective_labels]

        # create an empty array of -100 of length max_length
        argument_encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100
        effectiveness_encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * -100

        # set only labels whose first offset position is 0 and the second is not 0
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if mapping[0] == 0 and mapping[1] != 0:
                # overwrite label
                try:
                    argument_encoded_labels[idx] = argument_labels[i]
                    effectiveness_encoded_labels[idx] = effectiveness_labels[i]
                    i += 1
                except IndexError:
                    print("--- *The* TRY-Block ---")
                    print("Length of labels:" + str(len(argument_labels)))
                    print("Length of encoded_labels:" + str(len(argument_encoded_labels)))
                    print("IndexError with idx [" + str(idx) + "] and label number [" + str(i) + "]")
                    print("---")
                    continue

        # step 4: turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['argument_labels'] = torch.as_tensor(argument_encoded_labels)
        item['effectiveness_labels'] = torch.as_tensor(effectiveness_encoded_labels)

        return item

    def __len__(self):
        return self.len


def load_data(dataframe, batch_size):
    train_params = {'batch_size': batch_size,
                    'shuffle': True,
                    'num_workers': 0
                    }
    return DataLoader(dataframe, **train_params)


def model_train_multi_task(training_loader, model, optimizer, device, max_norm):

    loss_log = []

    tr_loss, tr_accuracy_taskA, tr_accuracy_taskB = 0, 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds_taskA, tr_labels_taskA = [], []
    tr_preds_taskB, tr_labels_taskB = [], []
    model.to(device)
    model.train()

    for idx, batch in enumerate(training_loader):
        ids = batch['input_ids'].to(device, dtype=torch.long)
        mask = batch['attention_mask'].to(device, dtype=torch.long)
        argument_labels = batch['argument_labels'].to(device, dtype=torch.long)
        effectiveness_labels = batch['effectiveness_labels'].to(device, dtype=torch.long)

        outputs = model(input_ids=ids, attention_mask=mask, taskA_labels=argument_labels,
                        taskB_labels=effectiveness_labels)
        loss = outputs.loss
        [logits_argument, logits_effectiveness] = outputs.logits
        tr_loss += loss.item()

        nb_tr_steps += 1
        nb_tr_examples += argument_labels.size(0)

        if idx % 100 == 0:
            loss_step = tr_loss / nb_tr_steps
            print("Training loss per 100 training steps: "+str(loss_step))
            loss_log.append(loss_step)

        # compute training accuracy
        flattened_targets_argument = argument_labels.view(-1)  # shape (batch_size * seq_len,)
        flattened_targets_effectiveness = effectiveness_labels.view(-1)  # shape (batch_size * seq_len,)
        active_logits_argument = logits_argument.view(-1,
                                                      model.num_labels_taskA)  # shape (batch_size * seq_len, num_labels)
        active_logits_effectiveness = logits_effectiveness.view(-1,
                                                                model.num_labels_taskB)  # shape (batch_size * seq_len, num_labels)

        flattened_predictions_argument = torch.argmax(active_logits_argument, axis=1)  # shape (batch_size * seq_len,)
        flattened_predictions_effectiveness = torch.argmax(active_logits_effectiveness,
                                                           axis=1)  # shape (batch_size * seq_len,)

        # only compute accuracy at active labels
        active_accuracy_argument = argument_labels.view(-1) != -100  # shape (batch_size, seq_len)
        active_accuracy_effectiveness = effectiveness_labels.view(-1) != -100  # shape (batch_size, seq_len)

        active_labels_argument = torch.where(active_accuracy_argument, argument_labels.view(-1),
                                             torch.tensor(-100).type_as(argument_labels))
        active_labels_effectiveness = torch.where(active_accuracy_effectiveness, effectiveness_labels.view(-1),
                                                  torch.tensor(-100).type_as(effectiveness_labels))

        labels_argument = torch.masked_select(flattened_targets_argument, active_accuracy_argument)
        labels_effectiveness = torch.masked_select(flattened_targets_effectiveness, active_accuracy_effectiveness)
        predictions_argument = torch.masked_select(flattened_predictions_argument, active_accuracy_argument)
        predictions_effectiveness = torch.masked_select(flattened_predictions_effectiveness,
                                                        active_accuracy_effectiveness)

        tr_labels_taskA.extend(labels_argument)
        tr_labels_taskB.extend(labels_effectiveness)

        tr_preds_taskA.extend(predictions_argument)
        tr_preds_taskB.extend(predictions_effectiveness)

        tmp_tr_accuracy_argument = accuracy_score(labels_argument.cpu().numpy(), predictions_argument.cpu().numpy())
        tmp_tr_accuracy_effectiveness = accuracy_score(labels_effectiveness.cpu().numpy(),
                                                       predictions_effectiveness.cpu().numpy())
        tr_accuracy_taskA += tmp_tr_accuracy_argument
        tr_accuracy_taskB += tmp_tr_accuracy_effectiveness

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=max_norm
        )

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy_argument = tr_accuracy_taskA / nb_tr_steps
    tr_accuracy_effectiveness = tr_accuracy_taskB / nb_tr_steps

    loss_log.append(epoch_loss)

    print(f"Training loss epoch: {epoch_loss}")
    print(f"Training accuracy epoch (argument): {tr_accuracy_argument}")
    print(f"Training accuracy epoch (effectiveness): {tr_accuracy_effectiveness}")

    return loss_log, tr_accuracy_argument, tr_accuracy_effectiveness


def get_sentence_predictions_multi_task(device, model, max_len, tokenizer, sentence):
    inputs = tokenizer(sentence,
                       is_split_into_words=True,
                       return_offsets_mapping=True,
                       padding='max_length',
                       truncation=True,
                       max_length=max_len,
                       return_tensors="pt")

    # print("TEST TOKENIZER")
    # inputs = tokenizer(["Ëin", "Wort"],
    #                     is_split_into_words=True,
    #                     # padding='max_length',
    #                     return_offsets_mapping=True,
    #                     truncation=True,
    #                     max_length=max_len,
    #                     return_tensors="pt")
    # ids = inputs["input_ids"].to(device)
    # tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
    # for token, mapping in zip(tokens, inputs["offset_mapping"].squeeze().tolist()):
    #     print(mapping, token)
    # print("...")
    # inputs = tokenizer(["Ein", "Wort"],
    #                     is_split_into_words=True,
    #                     # padding='max_length',
    #                     return_offsets_mapping=True,
    #                     truncation=True,
    #                     max_length=max_len,
    #                     return_tensors="pt")
    # ids = inputs["input_ids"].to(device)
    # tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
    # for token, mapping in zip(tokens, inputs["offset_mapping"].squeeze().tolist()):
    #     print(mapping, token)
    # sys.exit(0)

    # move to gpu
    ids = inputs["input_ids"].to(device)
    mask = inputs["attention_mask"].to(device)
    # forward pass
    outputs = model(ids, attention_mask=mask)
    [logits_argument, logits_effectiveness] = outputs.logits

    active_logits_argument = logits_argument.view(-1,
                                                  model.num_labels_taskA)  # shape (batch_size * seq_len, num_labels)
    active_logits_effectiveness = logits_effectiveness.view(-1,
                                                            model.num_labels_taskB)  # shape (batch_size * seq_len, num_labels)

    flattened_predictions_argument = torch.argmax(active_logits_argument, axis=1)  # shape (batch_size * seq_len,)
    flattened_predictions_effectiveness = torch.argmax(active_logits_effectiveness,
                                                       axis=1)  # shape (batch_size * seq_len,)

    tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
    token_predictions_argument = [Constants.ARGU_OUTPUT_LABELS[i] for i in flattened_predictions_argument.cpu().numpy()]
    token_predictions_effectiveness = [Constants.EFFECT_OUTPUT_LABELS[i] for i in
                                       flattened_predictions_effectiveness.cpu().numpy()]
    wp_preds_argument = list(
        zip(tokens, token_predictions_argument))  # list of tuples. Each tuple = (wordpiece, prediction)
    wp_preds_effectiveness = list(zip(tokens, token_predictions_effectiveness))
    
    # prediction_argument = ["O-Argument"] * len(inputs["offset_mapping"])
    prediction_argument = []
    for token_pred, mapping in zip(wp_preds_argument, inputs["offset_mapping"].squeeze().tolist()):

        # print(mapping, token_pred)
        if mapping[0] == 0 and mapping[1] != 0:
            prediction_argument.append(token_pred[1])
        else:
            continue

    # prediction_effectiveness = ["O-Effectiveness"] * len(inputs["offset_mapping"])
    prediction_effectiveness = []
    for token_pred, mapping in zip(wp_preds_effectiveness, inputs["offset_mapping"].squeeze().tolist()):
        if mapping[0] == 0 and mapping[1] != 0:
            prediction_effectiveness.append(token_pred[1])
        else:
            continue

    # If the last wordpiece is the end of the input sequence, there is no padding, i.e. the entire text was processed
    # But not everything fit into longformer: Fill remaining token positions with O prediction
    if (len(sentence) != len(prediction_argument)) and wp_preds_argument[-1][0] == "</s>":

        while(len(prediction_argument) < len(sentence)):
            prediction_argument.append("O-Argument")

    # If the last wordpiece is the end of the input sequence, there is no padding, i.e. the entire text was processed
    # But not everything fit into longformer: Fill remaining token positions with O prediction
    if (len(sentence) != len(prediction_effectiveness)) and wp_preds_effectiveness[-1][0] == "</s>":

        print("found one that does not fit")

        while(len(prediction_effectiveness) < len(sentence)):
            prediction_effectiveness.append("O-Effectivness")


    # To inspect length mismatches:
    # if (len(sentence) != len(prediction_argument)):

    #     preds_temp = []

    #     print(len(sentence), len(prediction_argument), len(prediction_effectiveness), len(wp_preds_argument), len(wp_preds_effectiveness))
    #     print(wp_preds_argument[-5:])
    #     for token_pred, mapping in zip(wp_preds_argument, inputs["offset_mapping"].squeeze().tolist()):

    #         # print(token_pred)
    #         if mapping[0] == 0 and mapping[1] != 0:
    #             # print(mapping, token_pred)
    #             # prediction_argument.append(token_pred[1])
    #             preds_temp.append(token_pred[0])
    #         else:
    #             continue

    #     print("---")
    #     print(preds_temp)
    #     print(sentence)
    #     for i in range(max(len(preds_temp), len(sentence))):
    #         print(preds_temp[i], sentence[i])  
    #     print()
    #     print("----")
    #     print()

    return prediction_argument, prediction_effectiveness


def get_sentence_predictions(device, model, max_len, tokenizer, sentence):
    inputs = tokenizer(sentence,
                       is_split_into_words=True,
                       return_offsets_mapping=True,
                       padding='max_length',
                       truncation=True,
                       max_length=max_len,
                       return_tensors="pt")

    # move to gpu
    ids = inputs["input_ids"].to(device)
    mask = inputs["attention_mask"].to(device)
    # forward pass
    outputs = model(ids, attention_mask=mask)
    logits = outputs[0]

    active_logits = logits.view(-1, model.num_labels)  # shape (batch_size * seq_len, num_labels)
    flattened_predictions = torch.argmax(active_logits,
                                         axis=1)  # shape (batch_size*seq_len,) - predictions at the token level

    tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
    token_predictions = [Constants.OUTPUT_LABELS[i] for i in flattened_predictions.cpu().numpy()]
    wp_preds = list(zip(tokens, token_predictions))  # list of tuples. Each tuple = (wordpiece, prediction)

    prediction = []
    for token_pred, mapping in zip(wp_preds, inputs["offset_mapping"].squeeze().tolist()):
        if mapping[0] == 0 and mapping[1] != 0:
            prediction.append(token_pred[1])
        else:
            continue
    return prediction


def get_final_prediction(test_text_df, y_pred, use_greater_10_condition=True):
    final_preds = []
    for i in tqdm(range(len(test_text_df))):
        idx = test_text_df.id.values[i]
        # print("---y_pred---")
        # print(y_pred)
        # print(len(y_pred))
        pred = [x.replace('B-', '').replace('I-', '') for x in y_pred[i]]
        # print("---pred---")
        # print(pred)
        # print(len(pred))
        j = 0
        while j < len(pred):
            cls = pred[j]
            if cls.startswith('O'):
                j += 1
            end = j + 1
            while end < len(pred) and pred[end] == cls:
                end += 1
            if use_greater_10_condition:
                if cls.startswith('O') is False and cls != '' and end - j > 10:
                    final_preds.append((idx, cls, ' '.join(map(str, list(range(j, end))))))
            else:
                if cls.startswith('O') is False and cls != '':
                    final_preds.append((idx, cls, ' '.join(map(str, list(range(j, end))))))

            j = end

    # print("---final preds arguments---")
    # print(final_preds)
    # print(len(final_preds))
    return final_preds

def get_final_prediction_effectiveness(test_text_df, y_pred, use_greater_10_condition=True):
    final_preds = []
    for i in tqdm(range(len(test_text_df))):
        idx = test_text_df.id.values[i]
        # print("---y_pred-e---")
        # print(y_pred)
        # print(len(y_pred))
        pred = y_pred[i]
        # print("---pred-e---")
        # print(pred)
        # print(len(pred))
        j = 0
        while j < len(pred):
            cls = pred[j]
            if cls.startswith('O'):
                j += 1
            end = j + 1
            while end < len(pred) and (not pred[end].startswith("B")) and pred[end].replace("B", "").replace("I", "") == cls.replace("B", "").replace("I", ""):
                end += 1
            if use_greater_10_condition:
                if cls.startswith('O') is False and cls != '' and end - j > 10:  # TODO: Checken, ob letzte Bedingung Sinn ergibt
                    final_preds.append((idx, cls[2:], ' '.join(map(str, list(range(j, end))))))
            else:
                if cls.startswith('O') is False and cls != '':
                    final_preds.append((idx, cls[2:], ' '.join(map(str, list(range(j, end))))))

            j = end

    # print("---final preds effectiveness---")
    # print(final_preds)
    # print(len(final_preds))
    return final_preds


def evaluate_token_accuracy(dataframe, pred_argument, pred_effectiveness):
    argument_acc = 0
    effectiveness_acc = 0

    argument_overall_pred = []
    effectiveness_overall_pred = []
    argument_overall_gold = []
    effectiveness_overall_gold = []

    for i in range(0, len(pred_argument)):
        discourse_gold = dataframe.iloc[i]['discourse_BIO']
        effectiveness_gold = dataframe.iloc[i]['effectiveness_BIO']
        current_arg_score = 0
        current_eff_score = 0
        try:
            current_arg_score = accuracy_score(discourse_gold, pred_argument[i])
            current_eff_score = accuracy_score(effectiveness_gold, pred_effectiveness[i])
            argument_overall_gold += discourse_gold
            argument_overall_pred += pred_argument[i]
            effectiveness_overall_gold += effectiveness_gold
            effectiveness_overall_pred += pred_effectiveness[i]
        except ValueError as e:
            print("argument", len(discourse_gold), len(pred_argument[i]))
            print("effectiveness", len(effectiveness_gold), len(pred_effectiveness[i]))
            # print(discourse_gold)
            # print(pred_argument[i])
            # print(effectiveness_gold)
            # print(pred_effectiveness[i])
            print(dataframe.iloc[i][1])
            print(str(e))
        argument_acc += current_arg_score
        effectiveness_acc += current_eff_score

    print('Argument elements token prediction accuracy: ', argument_acc/len(pred_argument))
    print('Argument elements token effectiveness accuracy: ', effectiveness_acc/len(pred_effectiveness))

    argument_accuracy = argument_acc/len(pred_argument)
    effectiveness_accuracy = effectiveness_acc/len(pred_effectiveness)

    # print(argument_accuracy)
    # print(effectiveness_accuracy)

    labels_effectiveness = Constants.EFFECT_OUTPUT_LABELS
    labels_argument = Constants.ARGU_OUTPUT_LABELS

    # cm_argument = confusion_matrix(y_true=argument_overall_gold, y_pred=argument_overall_pred, labels=labels_argument)
    # cm_argument = pd.DataFrame(cm_argument, index=labels_argument, columns=labels_argument)
    # cm_effectiveness = confusion_matrix(y_true=effectiveness_overall_gold, y_pred=effectiveness_overall_pred, labels=labels_effectiveness)
    # cm_effectiveness = pd.DataFrame(cm_effectiveness, index=labels_effectiveness, columns=labels_effectiveness)

    gold_argument = pd.Series(argument_overall_gold, name='Gold')
    pred_argument = pd.Series(argument_overall_pred, name='Predicted')
    cm_argument = pd.crosstab(gold_argument, pred_argument, margins=True)

    gold_effectiveness = pd.Series(effectiveness_overall_gold, name='Gold')
    pred_effectiveness = pd.Series(effectiveness_overall_pred, name='Predicted')
    cm_effectiveness = pd.crosstab(gold_effectiveness, pred_effectiveness, margins=True)

    # print("--- PREDICTIONS ---")
    # print(argument_overall_gold)
    # print(argument_overall_pred)
    # print()
    # print(effectiveness_overall_gold)
    # print(effectiveness_overall_pred)
    # print("---")

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     print(cm_effectiveness)
    #     print(cm_argument)

    return argument_accuracy, effectiveness_accuracy, cm_argument, cm_effectiveness


def model_predict_multi_task(device, model, max_len, tokenizer, dataframe, use_greater_10_condition=True):
    
    pred_argument = []
    pred_effectiveness = []
    for i, t in enumerate(dataframe['text_split'].tolist()):
        out_argument, out_effectiveness = get_sentence_predictions_multi_task(device, model, max_len, tokenizer, t)
        pred_argument.append(out_argument)
        pred_effectiveness.append(out_effectiveness)
    # print('predictions')
    # print(pred_argument)
    # print(pred_effectiveness)

    token_acc_argument, token_acc_effectiveness, cm_argument, cm_effectiveness = evaluate_token_accuracy(dataframe, pred_argument, pred_effectiveness)
    # print("ACCS", token_acc_argument, token_acc_effectiveness)

    pred_argument = pd.DataFrame(get_final_prediction(dataframe, pred_argument, use_greater_10_condition=use_greater_10_condition))
    # In case no prediction strings were returned
    if len(pred_argument) == 0:
        pred_argument = pd.DataFrame(columns=["essay_id", "discourse_type", "prediction_string"])
    else:
        pred_argument.columns = ["essay_id", "discourse_type", "prediction_string"]

    pred_effectiveness = pd.DataFrame(get_final_prediction_effectiveness(dataframe, pred_effectiveness, use_greater_10_condition=use_greater_10_condition))

    # In case no prediction strings were returned
    if len(pred_effectiveness) == 0:
        pred_effectiveness = pd.DataFrame(columns=["essay_id", "discourse_effectiveness", "prediction_string"])
    else:
        pred_effectiveness.columns = ["essay_id", "discourse_effectiveness", "prediction_string"]

    return pred_argument, pred_effectiveness, token_acc_argument, token_acc_effectiveness, cm_argument, cm_effectiveness


def get_confusion_matrix(df):

    gold = pd.Series(list(df['gold']), name='Gold')
    pred = pd.Series(list(df['pred']), name='Predicted')
    cm = pd.crosstab(gold, pred, margins=True)

    return cm


def calc_overlap2(set_pred, set_gt):
    """
    Calculates the overlap between prediction and
    ground truth and overlap percentages used for determining
    true positives.
    """
    # Length of each and intersection
    try:
        len_gt = len(set_gt)
        len_pred = len(set_pred)
        inter = len(set_gt & set_pred)
        overlap_1 = inter / len_gt
        overlap_2 = inter / len_pred
        return (overlap_1, overlap_2)
    except:  # at least one of the input is NaN
        return (0, 0)


def score_feedback_comp_micro(pred_df, gt_df, discourse_type, label):
    """
    A function that scores for the kaggle
        Student Writing Competition

    Uses the steps in the evaluation page here:
        https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    """
    scores = {}
    gt_df = gt_df.loc[gt_df[label] == discourse_type, ['essay_id', 'prediction_string']].reset_index(drop=True)
    pred_df = pred_df.loc[pred_df[label] == discourse_type, ['essay_id', 'prediction_string']].reset_index(drop=True)
    pred_df['pred_id'] = pred_df.index
    gt_df['gt_id'] = gt_df.index
    pred_df['prediction_string'] = [set(pred.split(' ')) for pred in pred_df['prediction_string']]
    gt_df['prediction_string'] = [set(pred.split(' ')) for pred in gt_df['prediction_string']]

    # Step 1. all ground truths and predictions for a given class are compared.
    joined = pred_df.merge(gt_df,
                           left_on='essay_id',
                           right_on='essay_id',
                           how='outer',
                           suffixes=('_pred', '_gt')
                           )
    overlaps = [calc_overlap2(*args) for args in zip(joined.prediction_string_pred, joined.prediction_string_gt)]

    # 2. If the overlap between the ground truth and prediction is >= 0.5,
    # and the overlap between the prediction and the ground truth >= 0.5,
    # the prediction is a match and considered a true positive.
    # If multiple matches exist, the match with the highest pair of overlaps is taken.
    joined['potential_TP'] = [(overlap[0] >= 0.5 and overlap[1] >= 0.5) for overlap in overlaps]
    joined['max_overlap'] = [max(*overlap) for overlap in overlaps]
    joined_tp = joined.query('potential_TP').reset_index(drop=True)
    tp_pred_ids = joined_tp.sort_values('max_overlap', ascending=False).groupby(['essay_id', 'gt_id'])['pred_id'].first()

    # 3. Any unmatched ground truths are false negatives
    # and any unmatched predictions are false positives.
    fp_pred_ids = set(joined['pred_id'].unique()) - set(tp_pred_ids)

    matched_gt_ids = joined_tp['gt_id'].unique()
    unmatched_gt_ids = set(joined['gt_id'].unique()) - set(matched_gt_ids)

    # Get numbers of each type
    TP = len(tp_pred_ids)
    scores['TP'] = TP
    FP = len(fp_pred_ids)
    scores['FP'] = FP
    FN = len(unmatched_gt_ids)
    scores['FN'] = FN

    if (TP + FN) != 0 and (TP + FP) != 0:
        scores['Precision'] = TP / (TP + FN)
        scores['Recall'] = TP / (TP + FP)
    # calc microf1
    my_f1_score = TP / (TP + 0.5 * (FP + FN))
    scores['F1'] = my_f1_score
    return scores


def score_feedback_comp_overall(gt_df, pred_df, label):
    """
    A function that scores for the kaggle
        Student Writing Competition

    Uses the steps in the evaluation page here:
        https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    """
   
    # MB: To later build a dataframe with discourse elements & their predictions (span-level) 
    init_value = 'MISSING'
    prediction_column = 'PRED_'+label
    prediction_id_column = 'PRED_ID'
    df_cm = deepcopy(gt_df.reset_index())
    df_cm['prediction_string'] = [set(pred.split(' ')) for pred in df_cm['prediction_string']]
    df_cm[prediction_column] = init_value
    df_cm[prediction_id_column] = init_value
    # MB: Save copy of pred
    pred_copy = deepcopy(pred_df.reset_index())
    pred_copy['prediction_string'] = [set(pred.split(' ')) for pred in pred_copy['prediction_string']]

    scores = {}
    gt_df = gt_df.loc[:, ['essay_id', 'prediction_string']].reset_index(drop=True)
    pred_df = pred_df.loc[:, ['essay_id', 'prediction_string']].reset_index(drop=True)
    pred_df['pred_id'] = pred_df.index
    gt_df['gt_id'] = gt_df.index
    pred_df['prediction_string'] = [set(pred.split(' ')) for pred in pred_df['prediction_string']]
    gt_df['prediction_string'] = [set(pred.split(' ')) for pred in gt_df['prediction_string']]

    # Step 1. all ground truths and predictions for a given class are compared.
    joined = pred_df.merge(gt_df,
                           left_on='essay_id',
                           right_on='essay_id',
                           how='outer',
                           suffixes=('_pred', '_gt')
                           )
    overlaps = [calc_overlap2(*args) for args in zip(joined.prediction_string_pred, joined.prediction_string_gt)]

    # 2. If the overlap between the ground truth and prediction is >= 0.5,
    # and the overlap between the prediction and the ground truth >= 0.5,
    # the prediction is a match and considered a true positive.
    # If multiple matches exist, the match with the highest pair of overlaps is taken.
    joined['potential_TP'] = [(overlap[0] >= 0.5 and overlap[1] >= 0.5) for overlap in overlaps]
    joined['max_overlap'] = [max(*overlap) for overlap in overlaps]
    joined_tp = joined.query('potential_TP').reset_index(drop=True)

    # For each essay and gt_id get pred_id with highest overlap:
    tp_pred_ids = joined_tp.sort_values('max_overlap', ascending=False).groupby(['essay_id', 'gt_id'])['pred_id'].first()
    # TODO MB: For these ids, get labels for cm (TPs)
    # already_seen = []
    for idx, tp_pred_id in tp_pred_ids.iteritems():

        tp_pred_id = int(tp_pred_id)

        # if tp_pred_id in already_seen:
        #     print("DUPLICATE", tp_pred_id)
        #     gt_ids = tp_pred_ids[tp_pred_ids == tp_pred_id]
        #     for gtid, predid in gt_ids.iteritems():
        #         print(gtid[1], predid)
        #         print(joined_tp[(joined_tp['gt_id']==gtid[1]) & (joined_tp['pred_id']==predid)])
        # already_seen.append(tp_pred_id)

        current_gt_id = idx[1]

        # MB: Sanity check to see if dfs are aligned
        element_copy = pred_copy.iloc[tp_pred_id]
        element_pred = pred_df[pred_df['pred_id']==tp_pred_id]
        same_ps = element_copy['prediction_string'] == element_pred['prediction_string']
        same_essay = element_copy['essay_id'] == element_pred['essay_id']
        # MB: Should only be one element
        if (len(same_ps) == 1) and (len(same_essay) == 1):
            same_ps = same_ps.iloc[0]
            same_essay = same_essay.iloc[0]
        else:
            print("Something went wrong when matching gt ids back to dataframe!")
            print(len(same_ps), len(same_essay))
            print(same_ps)
            print(same_essay)
            sys.exit(0)      
        if not (same_ps and same_essay):
            print("Misalignment between dataframes in evaluation!")
            print(element_copy)
            print(element_pred)
            sys.exit(0)
        
        # MB: Sanity checks for gold id
        element_cm = df_cm.iloc[current_gt_id]
        element_gt = gt_df[gt_df['gt_id']==current_gt_id]
        # MB: Remapping this id to the original dataframe should return the same element
        same_ps = element_cm['prediction_string'] == element_gt['prediction_string']
        same_essay = element_cm['essay_id'] == element_gt['essay_id']
        # MB: Should only be one element
        if (len(same_ps) == 1) and (len(same_essay) == 1):
            same_ps = same_ps.iloc[0]
            same_essay = same_essay.iloc[0]
        else:
            print("Something went wrong when matching gt ids back to dataframe!")
            print(len(same_ps), len(same_essay))
            print(same_ps)
            print(same_essay)
            sys.exit(0)      
        if not (same_ps and same_essay):
            print("Misalignment between dataframes in evaluation!")
            print(element_cm)
            print(element_gt)
            sys.exit(0)

        # MB: What was predicted for this pred id?
        prediction = pred_copy.loc[tp_pred_id, label]
        # MB: At position of matched gt_id
        df_cm.at[current_gt_id, prediction_column] = prediction
        # MB: What is the pred id at this point?
        df_cm.at[current_gt_id, prediction_id_column] = tp_pred_id

    # 3. Any unmatched ground truths are false negatives
    # and any unmatched predictions are false positives.
    # TODO MB: For any of these: pred has no match, becomes pred - NONE (FPs)
    # fp_pred_ids = set(joined['pred_id'].unique()) - set(tp_pred_ids)
    fp_pred_ids = set(pred_df['pred_id'].unique()) - set(tp_pred_ids)
    # MB: For each of the pred ids, append to df as pred - NONE
    for fp_pred_id in fp_pred_ids:
        # print(fp_pred_id)
        fp_pred_id = int(fp_pred_id)

        # MB: Sanity check: Does remapping return same element?
        element_copy = pred_copy.iloc[fp_pred_id]
        element_pred = pred_df[pred_df['pred_id']==fp_pred_id]
        same_ps = element_copy['prediction_string'] == element_pred['prediction_string']
        same_essay = element_copy['essay_id'] == element_pred['essay_id']
        # MB: Should only be one element
        if (len(same_ps) == 1) and (len(same_essay) == 1):
            same_ps = same_ps.iloc[0]
            same_essay = same_essay.iloc[0]
        else:
            print("Something went wrong when matching gt ids back to dataframe!")
            print(len(same_ps), len(same_essay))
            print(same_ps)
            print(same_essay)
            sys.exit(0)      
        if not (same_ps and same_essay):
            print("Misalignment between dataframes in evaluation!")
            print(element_copy)
            print(element_pred)
            sys.exit(0)

        # MB: What was predicted for this pred id?
        prediction = pred_copy.loc[fp_pred_id, label]
        # MB: concat this prediction and a gold_pred as None
        new_entry = pd.DataFrame(list(zip(['none'], [prediction], [fp_pred_id])), columns=[label, prediction_column, prediction_id_column])
        df_cm = pd.concat([df_cm, new_entry], ignore_index=True)
        # cm_pred.append(pred_df[pred_df['pred_id']==fp_pred_id][label+'_pred'])
        # cm_gold.append("None")

    # TODO MB: For any of these: gt has no match, becomes, NONE - gt (FNs)
    matched_gt_ids = joined_tp['gt_id'].unique()
    unmatched_gt_ids = set(joined['gt_id'].unique()) - set(matched_gt_ids)
    # MB: For each of these ids, put prediction in respective column of cm df as None
    for unmatched_gt_id in unmatched_gt_ids:
        element_cm = df_cm.iloc[unmatched_gt_id]
        element_gt = gt_df[gt_df['gt_id']==unmatched_gt_id]

        # MB: Sanity checks
        # MB: Remapping this id to the original dataframe should return the same element
        same_ps = element_cm['prediction_string'] == element_gt['prediction_string']
        same_essay = element_cm['essay_id'] == element_gt['essay_id']
        # MB: Should only be one element
        if (len(same_ps) == 1) and (len(same_essay) == 1):
            same_ps = same_ps.iloc[0]
            same_essay = same_essay.iloc[0]
        else:
            print("Something went wrong when matching gt ids back to dataframe!")
            print(len(same_ps), len(same_essay))
            print(same_ps)
            print(same_essay)
            sys.exit(0)      
        if not (same_ps and same_essay):
            print("Misalignment between dataframes in evaluation!")
            print(element_cm)
            print(element_gt)
            sys.exit(0)
        # MB: Now that we know the matching worked, set the prediction for this element to "NONE"
        # print("BEFORE", df_cm[df_cm.index==unmatched_gt_id]['PRED_'+label])
        # df_cm[df_cm.index==unmatched_gt_id]['PRED_'+label] = "none"
        df_cm.at[unmatched_gt_id, prediction_column] = "none"
        df_cm.at[unmatched_gt_id, prediction_id_column] = -1
        # print("AFTER", df_cm[df_cm.index==unmatched_gt_id]['PRED_'+label])
        # cm_gold.append(gold_df[gold_df['gt_id']==unmatched_gt_id][label+'_gt'])
        # cm_pred.append("None")

    # MB: Sanity check: There should be no 'MISSING' values in prediction column
    cm_value_counts = df_cm[prediction_column].value_counts()
    if init_value in cm_value_counts:
        print("Not all gold spans were assigned a prediction label!")
        sys.exit(0)
    # MB: Sanity check: The value_counts for df_pred and in prediction column should be the same
    # MB: Sanity check: No pred id should appear twice
    # MB: This cannot be fulfilled as some prediction spans are matched against two gold spans!
    pred_ids = df_cm[prediction_id_column].value_counts()
    # print(pred_ids)
    # MB debug: If an id occurs more than once, print all rows
    # MB: If an id occurs more than once, get all rows; if one of them has the correct prediction, keep that
    for pid in pred_ids.keys():
        # if pid != -1:
        #     if pred_ids[pid] == 2:
        #         delete_index = -1
        #         pid_rows = df_cm[df_cm[prediction_id_column]==pid]
        #         for idx, row in pid_rows.iterrows():
        #             if row[label] != row[prediction_column]:
        #                 delete_index = idx
        #         print(delete_index)
        #         # print(df_cm[df_cm[prediction_id_column]==pid].columns)
        #         # Is one of these the correct prediction?
        #         print(df_cm[df_cm[prediction_id_column]==pid][label])
        #         print(df_cm[df_cm[prediction_id_column]==pid][prediction_column])
        #         print()
        if (pid != -1) and (pred_ids[pid] > 2):
            print("There's more than two gold spans matched to one pred id!")
            sys.exit(0)

    # MB: Sanity check: Should have same number of predictions in both cm and original predictions
    # MB: Cannot be fulfilled, see above
    # orig_value_counts = pred_copy[label].value_counts()
    # print(orig_value_counts)
    # print(cm_value_counts)
    # for pred_label in orig_value_counts.keys():
    #     if not (orig_value_counts[pred_label] == cm_value_counts[pred_label]):
    #         print("Not the same number of predictions in original and confusion matrix!")
    #         print(pred_label, orig_value_counts[pred_label], cm_value_counts[pred_label])
    #         sys.exit(0)

    # Get numbers of each type
    TP = len(tp_pred_ids)
    scores['TP'] = TP
    FP = len(fp_pred_ids)
    scores['FP'] = FP
    FN = len(unmatched_gt_ids)
    scores['FN'] = FN

    if (TP + FN) != 0 and (TP + FP) != 0:
        scores['Precision'] = TP / (TP + FN)
        scores['Recall'] = TP / (TP + FP)
    # calc microf1
    my_f1_score = TP / (TP + 0.5 * (FP + FN))
    scores['F1'] = my_f1_score
    return scores, df_cm


def score_feedback_comp(gt_df, pred_df, label):

    # print("-- about to score feedback comp ---")
    # print("gold df")
    # print(gt_df.shape)
    # print(gt_df.columns)
    # print(gt_df.head(2))
    # print("pred df")
    # print(pred_df.shape)
    # print(pred_df.columns)
    # print(pred_df.head(2))
    # print("label", label)
    # print("------")

    scores = {}
    for sub_label in gt_df[label].unique():
        class_scores = score_feedback_comp_micro(pred_df, gt_df, sub_label, label)
        scores[sub_label] = class_scores

    overall_scores = {'TP': 0, 'FP': 0, 'FN': 0}
    f1_array = []
    for label in scores.keys():
        overall_scores['TP'] = overall_scores.get('TP') + scores.get(label).get('TP')
        overall_scores['FP'] = overall_scores.get('FP') + scores.get(label).get('FP')
        overall_scores['FN'] = overall_scores.get('FN') + scores.get(label).get('FN')
        f1_array.append(scores.get(label).get('F1'))
        print(label, "F1", scores.get(label).get('F1'))
        print(label, "TP", scores.get(label).get('TP'))
        print(label, "FP", scores.get(label).get('FP'))
        print(label, "FN", scores.get(label).get('FN'))
        print()

    overall_scores['Precision'] = overall_scores.get('TP') / (overall_scores.get('TP') + overall_scores.get('FN'))
    overall_scores['Recall'] = overall_scores.get('TP') / (overall_scores.get('TP') + overall_scores.get('FP'))
    overall_scores['F1'] = np.mean(f1_array)
    scores['overall'] = overall_scores

    print("Overall", "F1", scores.get('overall').get('F1'))
    print("Overall", 'TP:', overall_scores['TP'])
    print("Overall", 'FP:', overall_scores['FP'])
    print("Overall", 'FN:', overall_scores['FN'])

    f1 = scores.get('overall').get('F1')
    return f1, scores


def model_evaluate(data_pred, data_gold, label):
    col_list_gold = ["essay_id", label, "prediction_string"]
    # data_gold = data_gold[col_list_gold]

    # print(data_pred.columns)
    col_list_pred = ["essay_id", label, "prediction_string"]
    data_pred.reset_index(drop=True)
    data_pred = data_pred[col_list_pred]

    overall_f1, scores = score_feedback_comp(data_gold, data_pred, label)
    print("Overall F1 evaluation:", overall_f1)

    overall_scores, cm_df = score_feedback_comp_overall(data_gold, data_pred, label)

    gold = pd.Series(list(cm_df[label]), name='Gold')
    pred = pd.Series(list(cm_df["PRED_"+label]), name='Predicted')
    try:
        cm = pd.crosstab(gold, pred, margins=True)
    except:
        cm = pd.DataFrame()

    return overall_f1, scores, cm_df, cm


def write_evaluation(scores, output_path):
    df = pd.concat({k: pd.DataFrame(v) for k, v in scores.items()}, axis=1).stack(0).T
    df.to_csv(output_path, mode='a')


def write_prediction(pred, output_path):
    pred.to_csv(output_path, index=False)


def do_evaluation(experiment_name, setting_name, epoch, device, model, max_len, tokenizer, train, validate, test, train_df, validate_df, test_df, use_greater_10_condition=True):

    evaluation_output = {}
    
    try:
        print(f"Evaluate on train: {epoch + 1}")
        train_pred_argument, train_pred_effectiveness, train_token_acc_argument, train_token_acc_effectiveness, cm_token_argument_train, cm_token_effectiveness_train = model_predict_multi_task(device, model, max_len, tokenizer, train, use_greater_10_condition=use_greater_10_condition)
        
        cm_token_argument_train.to_csv(os.path.join(setting_name, str(epoch+1), "cm_train_token_argument_"+str(use_greater_10_condition)+".csv"), index_label='Gold|Predicted')    
        cm_token_effectiveness_train.to_csv(os.path.join(setting_name, str(epoch+1), "cm_train_token_effectiveness_"+str(use_greater_10_condition)+".csv"), index_label='Gold|Predicted')

        f1_argument_train, scores_argument_train, cm_span_argument_train_df, cm_span_argument_train = model_evaluate(train_pred_argument, train_df, "discourse_type")
        
        f1_effectiveness_train, scores_effectiveness_train, cm_span_effectiveness_train_df, cm_span_effectiveness_train = model_evaluate(train_pred_effectiveness, train_df,
                                                                "discourse_effectiveness")

        # Save prediction dataframes
        cm_span_argument_train_df.to_csv(os.path.join(setting_name, str(epoch+1), "predictions_train_span_argument_"+str(use_greater_10_condition)+".csv"))
        cm_span_effectiveness_train_df.to_csv(os.path.join(setting_name, str(epoch+1), "predictions_train_span_effectiveness_"+str(use_greater_10_condition)+".csv"))

        cm_span_argument_train.to_csv(os.path.join(setting_name, str(epoch+1), "cm_train_span_argument_"+str(use_greater_10_condition)+".csv"), index_label='Gold|Predicted')
        cm_span_effectiveness_train.to_csv(os.path.join(setting_name, str(epoch+1), "cm_train_span_effectiveness_"+str(use_greater_10_condition)+".csv"), index_label='Gold|Predicted')

        evaluation_output['train argument:' + str(epoch + 1)] = scores_argument_train
        evaluation_output['train effectiveness:' + str(epoch + 1)] = scores_effectiveness_train

        write_evaluation(evaluation_output, experiment_name + '/training_evaluation_epoch_' + str(epoch + 1) + "_" + str(use_greater_10_condition) + '.csv')
        write_prediction(train_pred_argument, experiment_name + '/training_argument_prediction_epoch_'+str(epoch + 1)+"_"+str(use_greater_10_condition)+'.csv')
        write_prediction(train_pred_effectiveness, experiment_name + '/training_effectiveness_prediction_epoch_' + str(epoch + 1) + "_"+str(use_greater_10_condition)+'.csv')
    except Exception as e:
        print(e)

    try:
        print(f"Validating epoch: {epoch + 1}")
        validate_pred_argument, validate_pred_effectiveness, val_token_acc_argument, val_token_acc_effectiveness, cm_token_argument_val, cm_token_effectiveness_val = model_predict_multi_task(device, model, max_len, tokenizer, validate, use_greater_10_condition=use_greater_10_condition)
        
        cm_token_argument_val.to_csv(os.path.join(setting_name, str(epoch+1), "cm_val_token_argument_"+str(use_greater_10_condition)+".csv"), index_label='Gold|Predicted')
        cm_token_effectiveness_val.to_csv(os.path.join(setting_name, str(epoch+1), "cm_val_token_effectiveness_"+str(use_greater_10_condition)+".csv"), index_label='Gold|Predicted')

        f1_argument_val, scores_argument_val, cm_span_argument_val_df, cm_span_argument_val = model_evaluate(validate_pred_argument, validate_df, "discourse_type")
        f1_effectiveness_val, scores_effectiveness_val, cm_span_effectiveness_val_df, cm_span_effectiveness_val = model_evaluate(validate_pred_effectiveness, validate_df,
                                                                "discourse_effectiveness")
        
        cm_span_argument_val_df.to_csv(os.path.join(setting_name, str(epoch+1), "predictions_val_span_argument_"+str(use_greater_10_condition)+".csv"))
        cm_span_effectiveness_val_df.to_csv(os.path.join(setting_name, str(epoch+1), "predictions_val_span_effectiveness_"+str(use_greater_10_condition)+".csv"))

        cm_span_argument_val.to_csv(os.path.join(setting_name, str(epoch+1), "cm_val_span_argument_"+str(use_greater_10_condition)+".csv"), index_label='Gold|Predicted')
        cm_span_effectiveness_val.to_csv(os.path.join(setting_name, str(epoch+1), "cm_val_span_effectiveness_"+str(use_greater_10_condition)+".csv"), index_label='Gold|Predicted')

        evaluation_output['validation argument:' + str(epoch + 1)] = scores_argument_val
        evaluation_output['validation effectiveness:' + str(epoch + 1)] = scores_effectiveness_val

        write_evaluation(evaluation_output, experiment_name + '/validation_evaluation_epoch_'+str(epoch + 1)+"_"+str(use_greater_10_condition) +'.csv')
        write_prediction(validate_pred_argument, experiment_name + '/validation_argument_prediction_epoch_'+str(epoch + 1)+"_"+str(use_greater_10_condition)+'.csv')
        write_prediction(validate_pred_effectiveness, experiment_name + '/validation_effectiveness_prediction_epoch_' + str(epoch + 1) + "_"+str(use_greater_10_condition)+'.csv')
    except Exception as e:
        print(e)

    # STEP 5: Test
    try:
        print("Test:")
        test_pred_argument, test_pred_effectiveness, test_token_acc_argument, test_token_acc_effectiveness, cm_token_argument_test, cm_token_effectiveness_test = model_predict_multi_task(device, model, max_len, tokenizer, test, use_greater_10_condition=use_greater_10_condition)

        cm_token_argument_test.to_csv(os.path.join(setting_name, str(epoch+1), "cm_test_token_argument_"+str(use_greater_10_condition)+".csv"), index_label='Gold|Predicted')
        cm_token_effectiveness_test.to_csv(os.path.join(setting_name, str(epoch+1), "cm_test_token_effectiveness_"+str(use_greater_10_condition)+".csv"), index_label='Gold|Predicted')
        
        f1_argument, scores_argument_test, cm_span_argument_test_df, cm_span_argument_test = model_evaluate(test_pred_argument, test_df, "discourse_type")
        f1_effectiveness, scores_effectiveness_test, cm_span_effectiveness_test_df, cm_span_effectiveness_test = model_evaluate(test_pred_effectiveness, test_df, "discourse_effectiveness")
        
        cm_span_argument_test_df.to_csv(os.path.join(setting_name, str(epoch+1), "predictions_test_span_argument_"+str(use_greater_10_condition)+".csv"))
        cm_span_effectiveness_test_df.to_csv(os.path.join(setting_name, str(epoch+1), "predictions_test_span_effectiveness_"+str(use_greater_10_condition)+".csv"))
        
        cm_span_argument_test.to_csv(os.path.join(setting_name, str(epoch+1), "cm_test_span_argument_"+str(use_greater_10_condition)+".csv"), index_label='Gold|Predicted')
        cm_span_effectiveness_test.to_csv(os.path.join(setting_name, str(epoch+1), "cm_test_span_effectiveness_"+str(use_greater_10_condition)+".csv"), index_label='Gold|Predicted')
                
        evaluation_output['test argument'] = scores_argument_test
        evaluation_output['test effectiveness'] = scores_effectiveness_test

        write_evaluation(evaluation_output, experiment_name + '/test_evaluation_epoch_'+str(epoch +1)+"_"+str(use_greater_10_condition)+".csv")
        write_prediction(test_pred_argument, experiment_name + '/test_argument_prediction_epoch'+str(epoch+1)+"_"+str(use_greater_10_condition)+'.csv')
        write_prediction(test_pred_effectiveness, experiment_name + '/test_effectiveness_prediction_epoch'+str(epoch+1)+"_"+str(use_greater_10_condition)+'.csv')
    except Exception as e:
        print(e)
    
    # Save token accuracy statistics
    token_acc_epoch = {"epoch": str(epoch+1), "train_argument": [train_token_acc_argument], "train_effectiveness": [train_token_acc_effectiveness], "val_argument": [val_token_acc_argument], "val_effectiveness": [val_token_acc_effectiveness], "test_argument": [test_token_acc_argument], "test_effectiveness": [test_token_acc_effectiveness]}
    # token_accuracy_overview = pd.concat([token_accuracy_overview, pd.DataFrame.from_dict(token_acc_epoch)])
    # token_accuracy_overview.to_csv(os.path.join(setting_name, "token_accuracy_stats.csv"), index=None)

    # Save span classification statistics

    # # Argument
    # df_span_argument_epoch = reduce(lambda  left,right: pd.merge(left,right,left_index=True, right_index=True, suffixes=['_train', '_val'],
                                        # how='outer'), [pd.DataFrame.from_dict(scores_argument_train, orient='index'), pd.DataFrame.from_dict(scores_argument_val, orient='index'), pd.DataFrame.from_dict(scores_argument_test, orient='index')])
    df_span_argument_epoch = pd.merge(left=pd.DataFrame.from_dict(scores_argument_train, orient='index'), right=pd.DataFrame.from_dict(scores_argument_val, orient='index'), left_index=True, right_index=True, suffixes=["_train", ""], how='outer')
    df_span_argument_epoch = pd.merge(left=df_span_argument_epoch, right=pd.DataFrame.from_dict(scores_argument_test, orient='index'), left_index=True, right_index=True, suffixes=["_val", "_test"], how='outer')
    df_span_argument_epoch["epoch"] = str(epoch+1)
    # span_argument_overview = pd.concat([span_argument_overview, df_span_argument_epoch])
    # span_argument_overview = span_argument_overview.fillna(0)
    # span_argument_overview.to_csv(os.path.join(setting_name, "span_argument_stats.csv"))

    # # Effectiveness
    df_span_effectiveness_epoch = pd.merge(left=pd.DataFrame.from_dict(scores_effectiveness_train, orient='index'), right=pd.DataFrame.from_dict(scores_effectiveness_val, orient='index'), left_index=True, right_index=True, suffixes=["_train", ""], how='outer')
    df_span_effectiveness_epoch = pd.merge(left=df_span_effectiveness_epoch, right=pd.DataFrame.from_dict(scores_effectiveness_test, orient='index'), left_index=True, right_index=True, suffixes=["_val", "_test"], how='outer')
    df_span_effectiveness_epoch["epoch"] = str(epoch+1)
    # span_effectiveness_overview = pd.concat([span_effectiveness_overview, df_span_effectiveness_epoch])
    # span_effectiveness_overview = span_effectiveness_overview.fillna(0)
    # span_effectiveness_overview.to_csv(os.path.join(setting_name, "span_effectiveness_stats.csv"))

    return token_acc_epoch, df_span_argument_epoch, df_span_effectiveness_epoch


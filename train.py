import tensorflow
import keras
import re
import json, pandas as pd
import numpy as np
from lib.query import Query
import torch
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
import numpy as np
from sklearn.metrics.classification import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, accuracy_score
from keras.preprocessing.text import text_to_word_sequence
from torch.nn.utils.rnn import pad_sequence
import pickle
import os
from nl2sql.model import *
from nl2sql.dataloader import DataTransformer
import nl2sql.eval_utils as eval_utils

os.chdir('/home/project')

train_sequences, train_columns_sequences, train_target = pickle.load(open('data/train_data.pic', 'rb'))
# pickle.dump([train_sequences, train_columns_sequences, train_target], open('data/train_data.pic', 'wb'))
test_sequences, test_columns_sequences, test_target = pickle.load(open('data/test_data.pic', 'rb'))
# pickle.dump([test_sequences, test_columns_sequences, test_target], open('data/test_data.pic', 'wb'))

# train_sequences, train_columns_sequences, train_target = train_sequences[:1000], train_columns_sequences[:1000], train_target[:1000]
# test_sequences, test_columns_sequences, test_target = test_sequences[:1000], test_columns_sequences[:1000], test_target[:1000]

datatransformer = pickle.load(open('data/datatransformerlemm.pic', 'rb'))

from torch.utils.data import DataLoader, Dataset

import pprint

pp = pprint.PrettyPrinter(indent=4)


class TrainingDataSet(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


# device = torch.device('cuda: 0')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
embedding_tensors = torch.tensor(datatransformer.embedding_matrix, device=device, dtype=torch.float)
print(embedding_tensors.shape)
train_sequences_tensor = torch.tensor(train_sequences, device=device, dtype=torch.long, requires_grad=False)
test_sequences_tensor = torch.tensor(test_sequences, device=device, dtype=torch.long, requires_grad=False)

test_columns_sequences_tensor = torch.tensor(test_columns_sequences, device=device, dtype=torch.long,
                                             requires_grad=False)
train_columns_sequences_tensor = torch.tensor(train_columns_sequences, device=device, dtype=torch.long,
                                              requires_grad=False)

train_target_tensor = torch.tensor(train_target, device=device, dtype=torch.long, requires_grad=False)
test_target_tensor = torch.tensor(test_target, device=device, dtype=torch.long, requires_grad=False)

train_dataset = TrainingDataSet(list(zip(train_sequences_tensor, train_columns_sequences_tensor)), train_target)
train_dataloader = DataLoader(train_dataset, batch_size=64)

test_dataset = TrainingDataSet(list(zip(test_sequences_tensor, test_columns_sequences_tensor)), test_target)
test_dataloader = DataLoader(test_dataset, batch_size=64)
# embedding_tensors.shape, embedding_matrix.shape

# pickle.dump(datatransformer, open('data/datatransformerlemm.pic', 'wb'))
model = NL2SQL.initialise_encoder_decoder_network(encoder_word_embedding_matrix=embedding_tensors,
                                                  encoder_max_columns_per_table=train_columns_sequences.shape[-2],
                                                  encoder_max_words_per_question=train_sequences.shape[-1],
                                                  encoder_n_lstm_cells=200,
                                                  encoder_bidirectional=True,
                                                  encoder_trainable_embedding=True,
                                                  encoder_n_layers=1,
                                                  decoder_n_lstm_cells=200,
                                                  decoder_n_layers=1,
                                                  decoder_op_seq_len=train_target.shape[-2],
                                                  decoder_action_embedding_dim=16,
                                                  decoder_bidirectional=True,
                                                  decoder_agg_ops=datatransformer.agg,
                                                  decoder_cond_ops=datatransformer.ops,
                                                  decoder_states=datatransformer.states_index)

model.cuda()
loss_function = nn.CrossEntropyLoss()
# loss_function = nn.CrossEntropyLoss(weight=torch.tensor(agg_weights, device=device, requires_grad=False, dtype=torch.float))
#     weight=torch.tensor([1, 2, 2, 2, 2, 2], dtype=torch.float, device=device,
#                                                         requires_grad=False))

optimizer = optim.Adam(model.parameters(), lr=1e-3)
# model = torch.load("data/training/NL2SQL15e:3.pt")
for i in range(100):
    model.train()
    loss_value = 0
    for batch_no, (X, y) in enumerate(train_dataloader):
        #         print(y.shape)
        batch_size, op_seq_len, n_actions = y.shape
        y_pred = model(*X, teacher_forcing_ratio=0.2, target_output_seq=torch.tensor(y,
                                                                                     dtype=torch.float,
                                                                                     device=device))
        y = y.view(-1, n_actions)
        # print(y_pred.shape)
        y_pred = y_pred.view(-1, n_actions)

        # _, y_true_label = torch.max(y, 1)
        y_true_label = y.argmax(dim=1)

        loss = loss_function(y_pred, y_true_label.cuda().long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value += loss.item()
        # print('\r', "Loss: {}, batch: {}".format(loss_value / (batch_no if batch_no > 0 else 1), batch_no), end="")
        print("Loss: {}, batch: {}".format(loss_value / (batch_no if batch_no > 0 else 1), batch_no), end="\n")
    # optimizer.zero_grad()
    # torch.save(model, 'data/training/NL2SQL15e:{epoch}.pt'.format(epoch=i))
    # model = torch.load("data/training/NL2SQL2e:0.pt")
    with torch.no_grad():
        model.eval()

        train_pred = []
        train_true = []

        test_loss_value = 0
        train_loss_value = 0

        confusion_matrix_test = []
        confusion_matrix_train = []

        for index, (X, y) in enumerate(train_dataloader):
            batch_size, op_seq_len, n_actions = y.shape
            y_pred_train = model(*X)
            y = y.view(-1, n_actions)
            # pickle.dump([y.tolist(), y_pred_train.tolist()], open('data/training/true_pred_ex.pic', 'wb'))
            # sys.exit()
            y_pred_train = y_pred_train.view(-1, n_actions)
            y_pred_train_label = y_pred_train.argmax(dim=1)
            y_true_train_label = y.argmax(dim=1)
            train_loss = loss_function(y_pred_train, y_true_train_label.cuda().long())
            train_loss_value += train_loss.item()
            confusion_matrix_ = confusion_matrix(y_true_train_label.cpu().numpy(),
                                                 y_pred_train_label.cpu().numpy(),
                                                 labels=np.arange(0, n_actions))
            confusion_matrix_train.append(confusion_matrix_)
            train_pred += y_pred_train_label.tolist()
            train_true += y_true_train_label.tolist()
            print("%d-train-eval" % index)

        train_loss_value = train_loss_value / index

        test_pred = []
        test_true = []
        test_pred_pd = []
        test_true_pd = []
        for index, (X, y) in enumerate(test_dataloader):
            batch_size, op_seq_len, n_actions = y.shape
            y_pred_test = model(*X)
            y = y.view(-1, n_actions)

            y_pred_test = y_pred_test.view(-1, n_actions)

            y_pred_test_label = y_pred_test.argmax(1)
            y_true_test_label = y.argmax(1)

            test_loss = loss_function(y_pred_test, y_true_test_label.cuda().long())
            test_loss_value += test_loss.item()

            confusion_matrix_ = confusion_matrix(y_true_test_label.cpu().numpy(),
                                                 y_pred_test_label.cpu().numpy(),
                                                 labels=np.arange(0, n_actions))

            confusion_matrix_test.append(confusion_matrix_)

            test_pred += y_pred_test_label.tolist()
            test_true += y_true_test_label.tolist()
            print("%d-test-eval" % index)

        test_loss_value = test_loss_value / index

    print("performance train/test loss: ", train_loss_value, test_loss_value)
    cm_train = np.sum(confusion_matrix_train, axis=0)
    cm_test = np.sum(confusion_matrix_test, axis=0)

    pickle.dump([cm_train, cm_test],
                open('data/training/reports/ConfusionMatrix|e:%d.pic' % i, 'wb'))

    torch.save(model, 'data/training/models/NL2SQL16e:{epoch}.pt'.format(epoch=i))

    #
    # test_report = eval_utils.get_slotwise_report(cm_test, 26, 44, 6, 4)
    #
    # train_report = eval_utils.get_slotwise_report(cm_train, 26, 44, 6, 4)

    test_report = eval_utils.get_slotwise_report(cm_test, datatransformer.max_words_per_question,
                                                 datatransformer.max_columns_per_table,
                                                 datatransformer.n_agg, datatransformer.n_ops)

    train_report = eval_utils.get_slotwise_report(cm_train, datatransformer.max_words_per_question,
                                                 datatransformer.max_columns_per_table,
                                                 datatransformer.n_agg, datatransformer.n_ops)

    test_report_df = pd.DataFrame.from_dict(test_report, orient='index')
    test_report_df.index = test_report_df.index.map(lambda x: 'testEpoch:%d_' % (i) + x)

    test_report_df.to_csv(open('data/training/reports/NL2SQL16Report.csv', 'a+'))

    train_report_df = pd.DataFrame.from_dict(train_report, orient='index')
    train_report_df.index = train_report_df.index.map(lambda x: 'trainEpoch:%d_' % (i) + x)

    train_report_df.to_csv(open('data/training/reports/NL2SQL16Report.csv', 'a+'))

    performance = {'test_report': test_report,
                   'train_report': train_report}

    pprint.pprint(performance, indent=4)
    # pprint.pprint(performance, open('data/trainAttention.log', 'a+'), indent=4)

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import tensorflow
import keras
import re
import json, pandas as pd
import numpy as np
# from lib.query import Query
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
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, accuracy_score
from keras.preprocessing.text import text_to_word_sequence
from torch.nn.utils.rnn import pad_sequence


class DataTransformer:
    """
    :param questions: list of questions
    :param headers: list of list of column names corresponding to the questions list
    :param sql_dicts: list of sql dictionary corresponding to the question list
    :param ops: list of ordered conditional operations s.t. ops at index i corresponds to the condops in sql_dict
    :param agg: list of aggeration operations s.t. agg at index i should correspond to the agg in sql_dict

    """

    def __init__(self, questions, header, sql_dict, ops, agg,
                 embedding_filepath='data/glove.6B.300d.txt',
                 max_words_per_question_percentile=99,
                 max_words_per_column_percentile=99,
                 max_columns_per_table_percentile=100):

        self.ops = ops
        self.agg = agg
        self.n_ops = len(ops)
        self.n_agg = len(agg)

        self.max_words_per_question_percentile = max_words_per_question_percentile
        self.max_words_per_column_percentile = max_words_per_column_percentile
        self.max_columns_per_table_percentile = max_columns_per_table_percentile
        self.lmtzr = WordNetLemmatizer()

        self.embedding_filepath = embedding_filepath
        self._fit(questions, header)
        self._initilize_index_of_seq()

        self.states_index = {'start': 0,
                             'agg': 1,
                             'selcol': 2,
                             'condcol': 3,
                             'condop': 4,
                             'condval': 5,
                             'end': 6}

        self.question_padding = 'pre'

    def clean_text(self, text):
        #         doc = nlp(text)
        #         text = ' '.join([token.lemma_ for token in doc])
        #         text = re.sub('\W+', ' ', text)
        text = text.lower()
        text = re.sub('\d+', 'NUMTAG', text)
        text = re.sub('=', ' equal to ', text)
        text = re.sub('<', '  less than ', text)
        text = re.sub('>', ' greater than ', text)
        text = ' '.join(list(map(self.lmtzr.lemmatize, word_tokenize(text))))
        #         text = re.sub('[\s]\W+[\s]', 'SYMNUM', text)
        return text

    @staticmethod
    def get_embedding_matrix(filepath):
        embeddings_index = {}
        EMBEDDING_DIM = int(re.findall(('(?<=\.)\d+(?=d)'), filepath)[0])
        f = open(filepath)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        print('Found %s word vectors.' % len(embeddings_index))

        word_index = {word: index + 2 for index, word in enumerate(embeddings_index.keys())}
        word_index['UNKWORD'] = 1
        #         print(list(word_index.items())[:10])

        index_word = {index: word for word, index in word_index.items()}

        embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                if embedding_vector.shape[0] == EMBEDDING_DIM:
                    embedding_matrix[i] = embedding_vector
                else:
                    print('incorrect dim', embedding_vector.shape, word)
            else:
                print("Not Found in Embedding Matrix: ", word)
        #                 pass
        return embedding_matrix, word_index, index_word

    def _fit(self, questions, headers):

        self.tokenizer = Tokenizer(filters='', oov_token='UNKWORD')
        self.embedding_matrix, word_index, index_word = DataTransformer.get_embedding_matrix(self.embedding_filepath)
        self.tokenizer.word_index, self.tokenizer.index_word = word_index, index_word

        #         self.tokenizer.fit_on_texts(list(map(self.clean_text, questions)) + [y for x in headers for y in x])
        #         self.tokenizer.fit_on_texts(list(questions) + [y for x in headers for y in x])
        question_sequences = self.tokenizer.texts_to_sequences(questions)
        columns_sequences = [self.tokenizer.texts_to_sequences(columns) for columns in headers]

        self.max_words_per_question = int(np.percentile(list(map(lambda x: len(x), question_sequences)),
                                                        self.max_words_per_question_percentile))

        self.max_words_per_column = int(np.percentile([len(col) for col_list in columns_sequences for col in col_list],
                                                      self.max_words_per_column_percentile))

        self.max_columns_per_table = int(np.percentile([len(col_list) for col_list in columns_sequences],
                                                       self.max_columns_per_table_percentile))

        #         self.embedding_matrix = DataTransformer.get_embedding_matrix(self.embedding_filepath, self.tokenizer.word_index)
        return self

    def transform_questions(self, questions):

        question_sequences = self.tokenizer.texts_to_sequences(list(map(self.clean_text, questions)))
        #         question_sequences = self.tokenizer.texts_to_sequences(list(questions))
        question_sequences = pad_sequences(question_sequences, maxlen=self.max_words_per_question, padding='pre',
                                           truncating='post')
        return question_sequences

    def transform_columns(self, columns):
        columns_sequences = [self.tokenizer.texts_to_sequences(columns_per_table) for columns_per_table in columns]
        columns_sequences = [pad_sequences(column_list, maxlen=self.max_words_per_column) for column_list in
                             columns_sequences]
        columns_sequences = DataTransformer.pad_columns(columns_sequences, maxlen=self.max_columns_per_table,
                                                        truncating='post', padding='post')
        return columns_sequences

    def ohe_agg(self, aggs):
        #         sel_col_ohe = np.zeros(max_columns_per_table)
        ohe_vector = np.zeros((len(aggs), self.n_agg))
        ohe_vector[np.arange(len(aggs)), aggs] = 1
        return ohe_vector

    def ohe_column(self, columns):
        ohe_vector = np.zeros((len(columns), self.max_columns_per_table))
        ohe_vector[np.arange(len(columns)), columns] = 1
        return ohe_vector

    def ohe_ops(self, ops):
        ohe_vector = np.zeros((len(ops), self.n_ops))
        ohe_vector[np.arange(len(ops)), ops] = 1
        return ohe_vector

    def _transform_conds(self, conditions):
        conditions = list(map(lambda x: self.clean_text(str(x)), conditions))
        return self.tokenizer.texts_to_sequences(conditions)

    def get_condition_seq(self, conds, questions):
        def find_start_end_index(masked):
            seq = ''.join(map(str, masked))
            ones_seq = re.findall('1+', seq)
            if not ones_seq:
                return None, None
            cond_seq = max(ones_seq, key=len)
            start_index = seq.index(cond_seq)
            end_index = start_index + len(cond_seq) - 1
            return start_index, end_index

        #         zipped = zip(self._transform_conds(conds), self.transform_questions(questions))
        masked = list(map(lambda cond, question: np.in1d(question, cond).astype(np.int),
                          self._transform_conds(conds), self.transform_questions(questions)))
        #         print(conds, questions)
        start_end_indices = np.array(list(map(find_start_end_index, masked)))
        start_seq, end_seq = np.zeros((len(questions), self.max_words_per_question)), np.zeros(
            (len(questions), self.max_words_per_question))

        for row, col in enumerate(start_end_indices[:, 0]):
            if col:
                start_seq[row, col] = 1
        for row, col in enumerate(start_end_indices[:, 1]):
            if col:
                end_seq[row, col] = 1

        return masked, start_seq, end_seq

    @staticmethod
    def pad_columns(list_of_list_cols, maxlen, dtype='int32', padding='post', truncating='post', value=0.0):

        def _pad_columns(sequences, maxlen, dtype, padding, truncating, value):
            diff = maxlen - len(sequences)
            len_of_each_seq = len(sequences[0])
            if diff > 0:
                if padding == 'pre':
                    return np.full(shape=(diff, len_of_each_seq), fill_value=value, dtype=dtype).tolist() + list(
                        sequences)
                else:
                    return list(sequences) + np.full(shape=(diff, len_of_each_seq), fill_value=value,
                                                     dtype=dtype).tolist()
            elif diff < 0:
                if truncating == 'pre':
                    return list(sequences[-diff:])
                else:
                    return list(sequences[:maxlen])
            return list(sequences)

        return np.array([_pad_columns(list_, maxlen, dtype, padding, truncating, value) for list_ in list_of_list_cols])

    def _initilize_index_of_seq(self):

        self.start_index = np.arange(0, 1)

        self.agg_index = np.arange(1, self.n_agg + 1)

        self.sel_index = np.arange(self.n_agg + 1, self.n_agg +
                                   self.max_columns_per_table + 1)
        self.concol_index = np.arange(self.n_agg +
                                      self.max_columns_per_table + 1, self.n_agg +
                                      self.max_columns_per_table +
                                      self.max_columns_per_table + 1)
        self.conops_index = np.arange(self.n_agg +
                                      self.max_columns_per_table +
                                      self.max_columns_per_table + 1, self.n_agg +
                                      self.max_columns_per_table +
                                      self.max_columns_per_table +
                                      self.n_ops + 1)
        self.condval_index = np.arange(self.n_agg +
                                       self.max_columns_per_table +
                                       self.max_columns_per_table +
                                       self.n_ops + 1, self.n_agg +
                                       self.max_columns_per_table +
                                       self.max_columns_per_table +
                                       self.n_ops +
                                       self.max_words_per_question + 1)
        self.end_index = np.arange(self.n_agg +
                                   self.max_columns_per_table +
                                   self.max_columns_per_table +
                                   self.n_ops +
                                   self.max_words_per_question + 1,
                                   self.n_agg +
                                   self.max_columns_per_table +
                                   self.max_columns_per_table +
                                   self.n_ops +
                                   self.max_words_per_question + 2)

        self.action_index = {'start': self.start_index,
                             'agg': self.agg_index,
                             'selcol': self.sel_index,
                             'condcol': self.concol_index,
                             'condops': self.conops_index,
                             'condval': self.condval_index,
                             'end': self.end_index}
        return self.action_index
        #         self.agg_index = np.arange(0, self.n_agg)
        #         self.sel_index = np.arange(self.n_agg, self.n_agg +
        #                                    self.max_columns_per_table)
        #         self.concol_index = np.arange(self.n_agg +
        #                                       self.max_columns_per_table, self.n_agg +
        #                                       self.max_columns_per_table +
        #                                       self.max_columns_per_table)
        #         self.conops_index = np.arange(self.n_agg +
        #                                       self.max_columns_per_table +
        #                                       self.max_columns_per_table, self.n_agg +
        #                                       self.max_columns_per_table +
        #                                       self.max_columns_per_table +
        #                                       self.n_ops)
        #         self.condval_index = np.arange(self.n_agg +
        #                                        self.max_columns_per_table +
        #                                        self.max_columns_per_table +
        #                                        self.n_ops, self.n_agg +
        #                                        self.max_columns_per_table +
        #                                        self.max_columns_per_table +
        #                                        self.n_ops +
        #                                        self.max_words_per_question)

        return self

    #     def tot_

    def reverse_label_sequence(self, seqs, threshold=0.5, questions=None):

        return list(map(lambda seq,
                               question: [self._reverse_label_sequence(one_seq, question) for one_seq in seq],
                        seqs,
                        questions))

    #     def _pad(self, seq, max_len, pre=True):
    def _normalize_question_index(self, index, question_len):

        if self.question_padding == 'pre':
            # this won't work on truncated question
            #             return list(map(lambda x: x - self.max_words_per_question + question_len, indices))
            return index - self.max_words_per_question + question_len
        if self.question_padding == 'post':
            return index

    @staticmethod
    def _remove_unconsecutive_indices(indices):
        if indices and len(indices) > 1:
            return [indices[i] for i in range(len(indices) - 1) if indices[i] == indices[i + 1] - 1] + [indices[-1]]
        return indices

    def _reverse_label_sequence(self, seq, question=None):

        seq = np.array(seq)
        index = np.argmax(seq)
        #         print(index)
        if index in self.start_index:
            return None, 0

        if index in self.agg_index:
            agg_seq = seq[self.agg_index]
            agg = np.argmax(agg_seq)
            state = 1
            return agg, state

        elif index in self.sel_index:
            sel_seq = seq[self.sel_index]
            sel = np.argmax(sel_seq)
            state = 2
            return sel, state

        elif index in self.concol_index:
            condcol_seq = seq[self.concol_index]
            condcol = np.argmax(condcol_seq)
            state = 3
            return condcol, state

        elif index in self.conops_index:
            condop_seq = seq[self.conops_index]
            condop = np.argmax(condop_seq)
            state = 4
            return condop, state

        elif index in self.condval_index:
            condval_seq = seq[self.condval_index]
            condval_index = np.argmax(condval_seq)

            condval = None
            if question:
                question_words = text_to_word_sequence(question,
                                                       filters=self.tokenizer.filters,
                                                       split=self.tokenizer.split,
                                                       lower=self.tokenizer.lower)
                #                 print(condval_index, question_words)

                condval_index = self._normalize_question_index(condval_index, len(question_words))
                #                 print(condval_index, question_words)
                condval = question_words[condval_index]

            state = 5
            return condval if condval else condval_index, state

        elif index in self.end_index:
            return None, 6

        #             condval = np.argwhere(condval_seq > threshold).squeeze()
        #             if question is not None:

        #                 question_words = text_to_word_sequence(question,
        #                                                     filters=self.tokenizer.filters,
        #                                                     split=self.tokenizer.split,
        #                                                     lower=self.tokenizer.lower)
        #                 if not hasattr(condval.tolist(), '__iter__'):
        #                     condval = [condval]
        # #                 print(question_words, condval, hasattr(condval, '__iter__'), list(condval))
        # #                 print(condval, question_words)
        #                 condval = self._normalize_question_index(condval, len(question_words))
        #                 condval = DataTransformer._remove_unconsecutive_indices(condval)

        #                 condvalwords = np.array(question_words)[condval]
        #                 condval = ' '.join(condvalwords)

        #             state = 5
        #             return condval, state

        return

    def pad_label_sequence(self, targets):

        def pad(x):
            #         print(max_len)
            n_pads = max_len - x.shape[0]
            #         print(n_pads, x.shape[0])
            pads = np.zeros((n_pads, x.shape[1]))
            pads[:, -1] = 1
            return np.concatenate((x, pads), axis=0)

        if not hasattr(self, "max_label_seq_len"):
            self.max_label_seq_len = max(targets, key=len).shape[0]
        max_len = self.max_label_seq_len
        n_actions = targets[0].shape[1]
        padded = map(lambda x: pad(x), targets)
        return np.array(list(padded))

    def label_sequence(self, sqls, questions):
        """
        iterate over list of sqls and questions
        """
        seq = map(lambda sql, question: self._label_sequence(sql, question), sqls, questions)

        targets = np.array(list(seq))
        return self.pad_label_sequence(targets)

    def _label_sequence(self, sql, question):
        """
        1-0-1 iteration over a sql and the corresponding question
        """

        def init_seq():
            seq = np.zeros(self.n_agg +
                           self.max_columns_per_table +
                           self.max_columns_per_table +
                           self.n_ops +
                           self.max_words_per_question + 2)
            return seq

        seq = []
        start = init_seq()
        end = init_seq()

        agg = init_seq()
        sel = init_seq()

        start[self.start_index] = 1
        end[self.end_index] = 1
        agg[self.agg_index] = self.ohe_agg([sql['agg']])
        sel[self.sel_index] = self.ohe_column([sql['sel']])

        seq.extend([start, agg, sel])

        for condcol, condops, condval in sql['conds']:
            condcol_seq = init_seq()
            condops_seq = init_seq()
            condval_start_seq = init_seq()
            condval_end_seq = init_seq()

            condcol_seq[self.concol_index] = self.ohe_column([condcol])
            condops_seq[self.conops_index] = self.ohe_ops([condops])
            #             _, start_index, end_index = self.get_condition_seq([condval], [question])
            _, condval_start_seq[self.condval_index], condval_end_seq[self.condval_index] = self.get_condition_seq(
                [condval], [question])
            seq.extend([condcol_seq, condops_seq, condval_start_seq, condval_end_seq])
        #         return seq
        seq.append(end)
        return np.array(seq)



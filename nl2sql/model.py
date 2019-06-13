import random

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, embedding_matrix,
                 max_columns_per_table,
                 max_words_per_question,
                 n_lstm_cells=200,
                 bidirectional=True,
                 trainable_embedding=True,
                 n_layers=1,
                 dropout=0.40):

        super(Encoder, self).__init__()
        vocab = embedding_matrix.shape[0]
        feature_dim = embedding_matrix.shape[1]

        self.word_embedding = nn.Embedding(vocab, feature_dim, _weight=embedding_matrix)

        self.word_embedding.weight.require_grad = trainable_embedding

        self.column_encoder = nn.LSTM(feature_dim,
                                      n_lstm_cells,
                                      num_layers=n_layers,
                                      bidirectional=bidirectional,
                                      dropout=dropout)

        # self.question_encoder = nn.LSTM(feature_dim,
        #                                 n_lstm_cells,
        #                                 num_layers=n_layers,
        #                                 bidirectional=bidirectional,
        #                                 dropout=dropout)

        self.question_encoder = self.column_encoder

        self.attended_question_encoder = nn.LSTM((n_lstm_cells * 2 if bidirectional else 1) + max_columns_per_table,
                                                 n_lstm_cells,
                                                 num_layers=n_layers,
                                                 bidirectional=bidirectional,
                                                 dropout=dropout)

        self.attended_column_encoder = nn.LSTM((n_lstm_cells * 2 if bidirectional else 1) + max_words_per_question,
                                               n_lstm_cells,
                                               num_layers=n_layers,
                                               bidirectional=bidirectional,
                                               dropout=dropout)

        self.n_layers = n_layers * 2 if bidirectional else n_layers

        self.n_lstm_cells = n_lstm_cells

        self.max_columns_per_table = max_columns_per_table
        self.max_words_per_question = max_words_per_question

        self.attention_1 = nn.Linear(self.n_layers * self.n_lstm_cells, (self.n_layers * self.n_lstm_cells) // 2)

        self.attention_2 = nn.Linear((self.n_layers * self.n_lstm_cells) // 2, 1)
        self.dropout = nn.Dropout(dropout)

    def init_hidden(self, batch_size):
        h_0, c_0 = [torch.zeros((self.n_layers, batch_size, self.n_lstm_cells), require_grad=False)] * 2
        return h_0, c_0

    def self_attention(self, output):
        #         output.shape() -> seq_len, batch_size, n_lstm_cells * n_layers
        output = output.transpose(1, 0, 2)
        atten_weights = self.attention_2(F.tanh(self.attention_1(output))).squeeze()
        #         atten_weights.shape() -> batch_size, seq_len
        atten_weights = F.softmax(atten_weights, dim=1)
        #         atten_weights.shape() -> batch_size, seq_len
        #         output.shape() -> batch_size, seq_len, n_lstm_cells * n_layers
        attended_output = torch.bmm(output.transpose(0, 2, 1), atten_weights.unsqueeze(2))
        #         attented_output.shape() -> batch_size, n_lstm_cells * n_layers
        return attended_output

    def alignment_attention(self, lstm_outputs, last_output):
        #         lstm_outputs.shape -> (batch_size, seq_len, hidden_size * n_layers)
        #         last_output.shape -> (batch_size, n_layers, hidden_size)
        lstm_outputs = lstm_outputs.permute(2, 1, 0)[-self.n_lstm_cells:].permute(2, 1, 0)
        #         lstm_outputs.shape -> (batch_size, seq_len, hidden_size)
        similarity = torch.bmm(lstm_outputs, last_output.permute(1, 0, 2)[-1].unsqueeze(2))
        #         similarity.shape -> (batch_size, seq_len, 1)

        atten_weights = F.softmax(similarity.squeeze(2), dim=1)

        attended = torch.bmm(atten_weights.unsqueeze(1), lstm_outputs).squeeze(2)
        #         attended.shape -> (batch_size, hidden_size)

        return attended

    def encode_columns(self, columns):
        batch_size, columns_per_table, words_per_column, embedding_dim = columns.shape
        columns = columns.view(-1, words_per_column, embedding_dim).permute(1, 0, 2)
        output, (ht, ct) = self.column_encoder(columns)
        # ht.shape -> n_layers, batch_size * columns_per_table, n_lstm_cells
        # output.shape -> words_per_column, batch_size * columns_per_table, n_dir * n_lstm_cells/hidden_size

        encoded_columns = output[-1].view(batch_size, columns_per_table, -1)
        # encoded_columns = ht.permute(1, 0, 2)[-1].view(batch_size, columns_per_table, -1)
        # encoded_columns.shape -> (batch_size, columns_per_table, hidden_size
        return encoded_columns

    #     def encode_questions(self, questions):
    #         output, (ht, ct) = self.question_encoder(questions)

    #         encoded_questions = self.alignment_attention(output.permute(1, 0, 2), ht.permute(1, 0, 2))

    def cross_attention(self, output_questions, output_columns):
        #         output_questions.shape -> (batch_size, seq_len, hidden_size)
        #         output_columns.shape -> (batch_size, columns_per_table, hidden_size)
        cross_attended = torch.bmm(output_questions, output_columns.permute(0, 2, 1))
        #         cross_attended.shape -> (batch_size, seq_len, columns_per_table)
        question_seq, column_seq = cross_attended, cross_attended.permute(0, 2, 1)
        return question_seq, column_seq

    def forward(self, questions, columns):
        questions = self.dropout(self.word_embedding(questions))
        columns = self.dropout(self.word_embedding(columns))
        questions_output, _ = self.question_encoder(questions.permute(1, 0, 2))
        questions_output = questions_output.permute(1, 0, 2)
        columns_output = self.encode_columns(columns)
        cross_attended_questions, cross_attended_columns = self.cross_attention(questions_output, columns_output)

        questions_cross_attended = torch.cat((questions_output, cross_attended_questions), dim=2)
        columns_cross_attended = torch.cat((columns_output, cross_attended_columns), dim=2)

        questions_output, _ = self.attended_question_encoder(questions_cross_attended.permute(1, 0, 2))

        columns_output, _ = self.attended_column_encoder(columns_cross_attended.permute(1, 0, 2))

        # lstm_output.shape -> seq_len, batch_size, n_dir * hidden_size
        # lstm_ht.shape -> n_layer * n_dir, batch_size, hidden_size

        # transposing to batch_first.
        questions_encoded = questions_output[-1]

        return questions_output.permute(1, 0, 2), columns_output.permute(1, 0, 2), questions_encoded


#         return questions_ht.permute(1, 0, 2), columns_ht.permute(1, 0, 2), columns_seq


class Decoder(nn.Module):
    def __init__(self, n_lstm_cells,
                 repr_dim,
                 n_layers,
                 op_seq_len,
                 action_embedding_dim,
                 bidirectional,
                 agg_ops,
                 cond_ops,
                 states,
                 use_self_attention=False,
                 dropout=0.2):
        """
        repr_dim -> hidden_dim of the encoded questions/columns
        op_seq_len -> max_length of the generated query

        """
        super(Decoder, self).__init__()
        #         self.action_embedding = nn.Embedding(len(actions), embedding_size)

        self.n_states = len(states) + len(cond_ops) + len(agg_ops) - 2

        self.action_embedding = nn.Embedding(self.n_states, action_embedding_dim)

        feature_dim = repr_dim + action_embedding_dim
        #         feature_dim += embedding_size
        self.decoder_lstm = nn.LSTM(feature_dim,
                                    n_lstm_cells,
                                    num_layers=n_layers,
                                    bidirectional=bidirectional, dropout=dropout)

        self.bilinear = nn.Bilinear(n_lstm_cells, repr_dim + action_embedding_dim, 1)

        self.feature_dim = feature_dim

        self.n_lstm_cells = n_lstm_cells
        self.n_layers = n_layers * 2 if bidirectional else n_layers

        self.agg_ops = agg_ops

        self.cond_ops = cond_ops

        self.start_idx = torch.arange(0, 1, device=device, dtype=torch.long)
        self.agg_idx = torch.arange(1, len(agg_ops) + 1, device=device, dtype=torch.long)
        self.selcol_idx = torch.arange(len(agg_ops) + 1, len(agg_ops) + 2, device=device, dtype=torch.long)
        self.condcol_idx = torch.arange(len(agg_ops) + 2, len(agg_ops) + 3, device=device, dtype=torch.long)
        self.condop_idx = torch.arange(len(agg_ops) + 3, len(agg_ops) + 3 + len(cond_ops), device=device,
                                       dtype=torch.long)
        self.condval_idx = torch.arange(len(agg_ops) + 3 + len(cond_ops), len(agg_ops) + 3 + len(cond_ops) + 1,
                                        device=device, dtype=torch.long)
        self.end_idx = torch.arange(len(agg_ops) + 3 + len(cond_ops) + 1, len(agg_ops) + 3 + len(cond_ops) + 2,
                                    device=device, dtype=torch.long)

        self.embedding_size = action_embedding_dim
        self.use_attention = use_self_attention
        self.op_seq_len = op_seq_len
        self.dropout = nn.Dropout(p=dropout)

    def generate_action_matrix(self, questions_encoded, columns_output, questions_output):
        # cols_repr_vector.shape -> batch_size, max_cols_per_tables, encoding_length
        # questions_repr_vector.shape -> batch_size, encoding_length
        # questions_output -> batch_size, max_words_per_question, endoding_length

        #         agg_idx = self.action_indices['agg']

        #         selcol_idx = self.action_indices['selcol']

        #         condcol_idx = self.action_indices['condcol']
        #         condop_idx = self.action_indices['condop']
        #         condval_idx = self.action_indices['condval']

        #         end_idx = self.actions['end']

        seq_len = questions_output.shape[-2]
        col_len = columns_output.shape[-2]
        hidden_size = questions_encoded.shape[-1]
        batch_size = questions_encoded.shape[0]

        # padding_zeros shape -> 1, hiddent_size of question/column encoding
        padding_zeros = torch.zeros(1, hidden_size, device=device, requires_grad=False)

        # start_matrix shape -> 1, hidden_size of action_embedding
        start_vector = torch.cat((self.action_embedding(self.start_idx), padding_zeros), dim=1).repeat(batch_size, 1, 1)

        # agg_matrix shape -> n_ops, hiddden_size
        agg_vector = torch.cat((self.action_embedding(self.agg_idx),
                                padding_zeros.repeat(self.agg_idx.shape[0], 1)), dim=1).repeat(batch_size, 1, 1)

        #
        selcol_vector = torch.cat((self.action_embedding(self.selcol_idx).repeat(batch_size, col_len, 1),
                                   columns_output), dim=2)

        concol_vector = torch.cat((self.action_embedding(self.condcol_idx).repeat(batch_size, col_len, 1),
                                   columns_output), dim=2)

        condops_vector = torch.cat((self.action_embedding(self.condop_idx),
                                    padding_zeros.repeat(self.condop_idx.shape[0], 1)), dim=1).repeat(batch_size, 1, 1)

        condval_vector = torch.cat((self.action_embedding(self.condval_idx).repeat(batch_size, seq_len, 1),
                                    questions_output), dim=2)

        end_vector = torch.cat((self.action_embedding(self.end_idx), padding_zeros),
                               dim=1).repeat(batch_size, 1, 1)

        all_actions_matrix = torch.cat([start_vector,
                                        agg_vector,
                                        selcol_vector,
                                        concol_vector,
                                        condops_vector,
                                        condval_vector,
                                        end_vector], dim=1)
        # that was tough

        return all_actions_matrix

    #     def get_action_vector(self, output_seq, n_words_per_question, n_columns_per_table):
    #         index = output_seq.argmax()

    # #         {'start_idx' : np.arange(0, 1),
    # #         'agg_idx':  np.arange(1, )}

    #         # for start and agg index
    #         if index < 1 + len(self.agg_ops):
    #             return self.action_embedding.weight[index]

    #         # selcol
    #         elif index < 1 + len(self.agg_ops) + n_columns_per_table:
    #             return self.action_embedding.weight[1 + len(self.agg_ops)]

    #         # condcols
    #         elif index <  1 + len(self.agg_ops) + 2 * n_columns_per_table:
    #             index = 1 + len(self.agg_ops) + 1
    #             return self.action_embedding.weight[index]

    #         # condops
    #         elif index <  1 + len(self.agg_ops) + 2 * n_columns_per_table + len(self.cond_ops):
    #             index = index - (1 + len(self.agg_ops) + 2 * n_columns_per_table)  + (1 + len(self.agg_ops) + 2)
    #             return self.action_embedding.weight[index]

    #         # condval
    #         elif index <  1 + len(self.agg_ops) + 2 * n_columns_per_table + len(self.cond_ops) + n_words_per_question:
    #             index = 1 + len(self.agg_ops) + 2 + len(self.cond_ops)
    #             return self.action_embedding.weight[index]

    #         # end
    #         elif index <  1 + len(self.agg_ops) + 2 * n_columns_per_table + len(self.cond_ops) + n_words_per_question + 1:
    #             index =  1 + len(self.agg_ops) + 2 + len(self.cond_ops) + 1
    #             return self.action_embedding.weight[index]

    def global_attention(questions_output, target):
        pass

    def get_action_vector_from_output(self, output_seq, action_matrix):
        """
        output_seq.shape -> batch_size, n_actions
        action_matrix.shape -> batch_size, n_actions, hidden_size
        """
        top_index = torch.argmax(output_seq, dim=1).detach()
        actions = torch.cat([action_matrix[n_batch_index, action_index, :self.embedding_size].unsqueeze(0).detach()
                             for n_batch_index, action_index in enumerate(top_index)], dim=0)

        #         action_matrix.shape -> batch_size, self.embedding_size/action_embedding_dim
        return actions

    # def get_action_vector_from_output(self, output_seq, action_matrix):
    #     """
    #     output_seq.shape -> batch_size, n_actions
    #     action_matrix.shape -> batch_size, n_actions, hidden_size
    #     """
    #     top_index = torch.argmax(output_seq, dim=1).detach()
    #     actions = torch.cat([action_matrix[n_batch_index, action_index].unsqueeze(0).detach()
    #                          for n_batch_index, action_index in enumerate(top_index)], dim=0)
    #
    #     #         action_matrix.shape -> batch_size, self.embedding_size/action_embedding_dim
    #     return actions

    def forward_step(self, previous_action_vector, questions_encoded, previous_hidden, output_actions_matrix):
        """
        takes the previous_state batch and predicts the next token
        """
        # previous_action_vector.shape -> batch_size, action_embedding_dim
        # questions_encoded.shape -> batch_size, hidden_dim
        # output_actions_matrix.shape -> batch_size, n_possible_actions, hidden_size
        #         question_encoded = self.question_encoder()

        n_actions_outputs = output_actions_matrix.shape[-2]

        decoder_ip = torch.cat((previous_action_vector, questions_encoded), dim=1)

        decoder_ip = decoder_ip.unsqueeze(1).permute(1, 0, 2)
        _, hidden_states = self.decoder_lstm(decoder_ip, previous_hidden)
        last_output = hidden_states[0][-1]
        # last_output.shape -> batch_size, hidden_size/decoder_n_lstm_cells
        last_output = last_output.unsqueeze(1).repeat(1, n_actions_outputs, 1)
        # last_ouput.shape -> batch_size, n_actions_outputs, hidden_size

        bilinear_output = self.bilinear(last_output, output_actions_matrix)
        # bilinear_output.shape -> batch_size, n_actions_ouputs, 1
        bilinear_output = bilinear_output.squeeze(2)

        return bilinear_output, hidden_states

    def generate_hidden(self, batch_size):
        n_layers = self.n_layers
        h_0 = torch.zeros(n_layers, batch_size, self.n_lstm_cells, requires_grad=False, device=device)
        c_0 = torch.zeros(n_layers, batch_size, self.n_lstm_cells, requires_grad=False, device=device)
        return h_0, c_0

    def forward(self, questions_encoded, questions_output, columns_output, teacher_forcing_ratio=0,
                target_output_seq=None):
        """
        questions_encoded.shape -> batch_size, hidden_size | question representation
        columns_output_vector.shape -> batch_size, max_columns_per_table, hidden_size | column representation
        questions_output.shape -> batch_size, max_word_per_table, hidden_size | question representation at word level, used for attention
        """

        batch_size = questions_encoded.shape[0]
        # Prediction for start not required, start from index = 1
        previous_hidden = self.generate_hidden(batch_size)

        # start action
        previous_action = self.action_embedding.weight[0].repeat(batch_size, 1)
        action_matrix = self.generate_action_matrix(questions_encoded, columns_output, questions_output)
        # previous_action = action_matrix[:, 0, :]
        # action_matrix.shape -> batch_size, n_actions, action_embedding_size(+repr_dim)
        start_seq = torch.zeros((batch_size, 1, action_matrix.shape[-2]), device=device)
        start_seq[:, :, 0] = 1

        output_seq_list = [start_seq]

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if not use_teacher_forcing:
            for index in range(1, self.op_seq_len):
                output_seq, previous_hidden = self.forward_step(previous_action, questions_encoded, previous_hidden,
                                                                action_matrix)
                previous_action = self.get_action_vector_from_output(output_seq, action_matrix)
                output_seq_list.append(output_seq.unsqueeze(1))
        else:
            for index in range(1, self.op_seq_len):
                output_seq, previous_hidden = self.forward_step(previous_action, questions_encoded, previous_hidden,
                                                                action_matrix)
                #                 output_seq.shape -> batch_size, op_seq_len
                previous_action = self.get_action_vector_from_output(target_output_seq[:, index - 1, :], action_matrix)
                output_seq_list.append(output_seq.unsqueeze(1))

        out_seqs = torch.cat(output_seq_list, dim=1)
        # out_seqs.shape -> batch_size, op_seq_len, n_actions

        return out_seqs


class NL2SQL(nn.Module):

    def __init__(self, encoder, decoder, ):
        super(NL2SQL, self).__init__()
        self.decoder = decoder
        self.encoder = encoder

    @classmethod
    def initialise_encoder_decoder_network(cls, encoder_word_embedding_matrix,
                                           encoder_max_columns_per_table,
                                           encoder_max_words_per_question,
                                           encoder_n_lstm_cells,
                                           encoder_bidirectional,
                                           encoder_trainable_embedding,
                                           encoder_n_layers,
                                           decoder_n_lstm_cells,
                                           decoder_n_layers,
                                           decoder_op_seq_len,
                                           decoder_action_embedding_dim,
                                           decoder_bidirectional,
                                           decoder_agg_ops,
                                           decoder_cond_ops,
                                           decoder_states, ):
        encoder = Encoder(embedding_matrix=encoder_word_embedding_matrix,
                          max_columns_per_table=encoder_max_columns_per_table,
                          max_words_per_question=encoder_max_words_per_question,
                          n_lstm_cells=encoder_n_lstm_cells,
                          bidirectional=encoder_bidirectional,
                          trainable_embedding=encoder_trainable_embedding,
                          n_layers=encoder_n_layers)
        decoder = Decoder(n_lstm_cells=decoder_n_lstm_cells,
                          repr_dim=encoder_n_lstm_cells * 2 if encoder_bidirectional else 1,
                          n_layers=decoder_n_layers,
                          op_seq_len=decoder_op_seq_len,
                          action_embedding_dim=decoder_action_embedding_dim,
                          bidirectional=decoder_bidirectional,
                          agg_ops=decoder_agg_ops,
                          cond_ops=decoder_cond_ops,
                          states=decoder_states,
                          use_self_attention=False)

        return cls(encoder, decoder)

    def flatten_parameters(self):
        self.encoder.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, questions, columns, teacher_forcing_ratio=0., target_output_seq=None):
        questions_output, columns_output, questions_encoded = self.encoder(questions, columns)
        out_seqs = self.decoder(questions_encoded, questions_output,
                                columns_output, teacher_forcing_ratio, target_output_seq)
        return out_seqs



if __name__ == "__main___":
    pass

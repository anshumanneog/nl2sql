import torch
import re
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SequencePredictor:
    def __init__(self, model, datatransformer):
        self.model = model
        self.datatransformer = datatransformer
        self.model.eval()

    @classmethod
    def fromFile(cls, model_filepath, datatransformer_filepath, *args, **kwargs):
        datatransformer = pickle.load(datatransformer_filepath)
        model = torch.load(model_filepath)
        return cls(model, datatransformer, *args, **kwargs)

    # @staticmethod
    def format_sql(self, pred_seqs, questions_words, columns_words):
        queries = []
        for pred_seq, question_words, column_words in zip(pred_seqs, questions_words, columns_words):
            query = "SELECT"
            agg_select_end = False
            cond_end = False
            start = None
            for seq, state in pred_seq:

                if state == 0:
                    token = ''
                elif state == 1:
                    token = self.datatransformer.agg[seq]

                elif state == 2:
                    # print(state, seq)
                    token = '(' + column_words[seq] + ')'
                    agg_select_end = True

                elif state == 3:
                    token = column_words[seq]
                    if agg_select_end:
                        token = 'FROM {table_name} WHERE '.format(table_name='t1') + token
                        agg_select_end = False
                    if cond_end:
                        token = 'AND ' + token
                        cond_end = False

                elif state == 4:
                    token = self.datatransformer.ops[seq]

                elif state == 5:
                    if start:
                        if start != seq:
                            token = re.search('{end}.*?{start}'.format(start=start[::-1], end=seq[::-1]),
                                              question_words.lower()[::-1])
                            token = token.group()[::-1] if token else ''
                        else:
                            token = seq
                        cond_end = True
                        start = None
                    else:
                        start = seq
                        token = ''
                elif state == 6:
                    token = ''

                if start and state!=5:
                    token = start + token
                    start = None
                    cond_end = True
                if token:
                    query = query + ' ' + token


            queries.append(query)
        return queries

    def predict(self, questions_words, columns_words):
        questions = self.datatransformer.transform_questions(questions_words)
        columns = self.datatransformer.transform_columns(columns_words)
        questions = torch.tensor(questions, dtype=torch.long, device=device)
        columns = torch.tensor(columns, dtype=torch.long, device=device)
        pred_seqs = self.model(questions, columns)

        pred_seqs = self.datatransformer.reverse_label_sequence(pred_seqs.detach().cpu().numpy(),
                                                                questions=questions_words)
        print(pred_seqs)

        return self.format_sql(pred_seqs,
                               questions_words=questions_words,
                               columns_words=columns_words)

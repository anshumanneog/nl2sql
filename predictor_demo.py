import torch
from nl2sql.dataloader import DataTransformer
from nl2sql.predictor import SequencePredictor
from nl2sql.model import NL2SQL
import os
import pickle
from nl2sql.predictor import SequencePredictor
import pandas as pd
import re


def format_sql(sels, aggs):
    sql = "SELECT {agg}({col}) FROM t1"
    return [sql.format(agg=agg, col=sel) for agg, sel in zip(aggs, sels)]


def sub_find(x):
    x = re.sub('franchisee', 'franchise', x, flags=re.IGNORECASE)
    return x


def sub_cols(xs):
    xs = list(map(lambda x: re.sub('ACCT', 'account', x, flags=re.IGNORECASE), xs))
    return xs


df = pd.read_csv('Sony_test_data.csv')
df.header = df.header.apply(lambda x: sub_cols(eval(x)))

os.chdir('/home/project')
datatransformer = pickle.load(open('data/datatransformer_all.pic', 'rb'))
model = torch.load("data/training/models/NL2SQL17e:11.pt")

predictor = SequencePredictor(model, datatransformer)

questions = ['Find the Distinct count of game names for RPG genre',
       'Find the number of games for RPG and Adventure genre',
       'Find the number of distinct Title_ID for Horizon Zero Dawn',
       'Find the Number of games under franchise 125690',
       'FIND TOTAL PLAYTIME FOR GAME_NAME_ID = 123456',
       'FIND TOTAL NUMBER OF ACCOUNTS WHO PLAYED GAME_NAME_ID = 123456',
       'FIND TOTAL NUMBER OF ACCOUNTS WHO PLAYED ON DATE = 11/20/2018',
       'Find total revenue for the title whose title_id=118444 in US.  ',
       'Find number of accounts in region 4 ',
       'Find total sales for the title whose title_id=118444',
       'Count number of accounts of who made transaction on 11/26/2018.',
       'How many distinct trophy id are there for trophy name Battle for Control',
       "List of trophies that are marked under title is 13333",
       'What is the trophy_type of trophy name "The Ballerina"',
       'How many franchise names are present for franchise name id 75230',
       'How many distinct Game names are present for franchise name id 75230',
       'How many franchise names are present for Party Type 3 rd Party',
       'What is/are the game genre(s) for franchise name id 75230']
# questions = list(map(sub_find, df.question))
headers = df.header

preds = predictor.predict(questions, headers)
import pprint
pprint.pprint(dict(zip(questions, preds)))

preds = predictor.predict(['how is the highest paid salary  ?',
                                  'Number of employees salary less than 1000 but greater than 2000?',
                          # 'who is the fighter with no losses in wwe ?',
                          'what notes are used in Australia ? ',
                          # 'what is the minimum pay?',
                          'what is the average compensation? ',
                         'what is the highest electrons ?'],
                    [['employee_ID', 'salary', 'age', 'sex', 'designation', 'date of joining', 'address'],
                     ['employee_ID', 'salary', 'age', 'sex', 'designation', 'date of joining', 'address'],
                     # ['fighter_id', 'weight', 'wins', 'losses', 'origin'],
                    ['currency', 'US dollar value', 'country'],
                    # ['no', 'salary', 'age', 'sex', 'designation', 'date of joining', 'address'],
                    ['no', 'salary', 'age', 'sex', 'designation', 'date of joining', 'address'],
                    ['id', 'element', 'relative atomic mass', 'atomic number']])


print(preds)
# print(format_sql(*preds))
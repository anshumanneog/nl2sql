Query.agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG', 'COUNT_DISTINCT']


class load_data:
    def __init__():
        pass


def inject_table_names(sql, table_cols):
    """
        replace col<index> in `sql` query with the <column_name> from `table_cols`
        """
    try:
        regex = re.compile(r'\scol(\d+)')
        return regex.sub(lambda x: ' ' + table_cols[int(x.string[x.start() + 4: x.end()])], str(sql))
    except Exception as e:
        print(e, sql, table_cols)
    #         pass
    return None


# def clean_text(text):
#     return 
#     text = re.sub('\W+', ' ', text)
#     return text


def load_data(filepath_phrase, filepath_table):
    lines = extra_q
    lines += open(filepath_phrase, 'r').readlines()
    df_phrases = pd.DataFrame([json.loads(line) for line in lines])

    lines = extra_col
    lines += open(filepath_table, 'r').readlines()
    df_tables = pd.DataFrame([json.loads(line) for line in lines])
    # join table data with sql data
    df = pd.merge(df_phrases, df_tables[['id', 'header']], left_on='table_id', right_on='id')
    #     df= df[:10000]
    df['query_temp'] = df.sql.apply(lambda data: Query.from_dict(d=data))
    df['query'] = df.loc[:, ['query_temp', 'header']] \
        .apply(lambda row: inject_table_names(row[0], row[1]), axis=1)
    df['agg'] = df['sql'].apply(lambda x: x['agg'])
    df['sel_col'] = df['sql'].apply(lambda x: x['sel'])

    #     questions = df['question'].apply(clean_text).values
    questions = df['question'].values
    queries = df['query'].values
    column_names = df['header'].values
    agg = df['agg'].values
    sel_col = df['sel_col'].values
    return questions, queries, column_names, df, agg, sel_col


test_questions, test_queries, test_column_names, df_test, test_agg, test_sel_col = load_data('data/test.jsonl',
                                                                                             'data/test.tables.jsonl')
train_questions, train_queries, train_column_names, df_train, train_agg, train_sel_col, = load_data('data/train.jsonl',
                                                                                                    'data/train.tables.jsonl')

dev_questions, dev_queries, dev_column_names, _, dev_agg, dev_sel_col = load_data('data/dev.jsonl',
                                                                                  'data/dev.tables.jsonl')
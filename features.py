import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import scipy as sp
from scipy.sparse import coo_matrix, hstack, vstack
import json


def get_dataframe(train_loc: str, test_loc: str):
    
    df_train = pd.read_json(train_loc,lines=True)
    df_test = pd.read_json(test_loc,lines=True)
    return df_train, df_test

def update_text(df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:

    for df in [df_train, df_test]:
        Pro_texts = []
        Con_texes = []
        for debate in df['rounds'].tolist():
            Pro_text = ""
            Con_text = ""
            for r, round in enumerate(debate):
                Pro_text += "\n --------------------------------------------------" + str(r) + "--------------------------------------------------"
                Con_text += "\n --------------------------------------------------" + str(r) + "--------------------------------------------------"
                for log in round:
                    if log['side'] == 'Pro':
                        Pro_text += log['text']
                    else:
                        Con_text += log['text']

            Pro_texts.append(Pro_text)
            Con_texes.append(Con_text)

        df['Pro_text'] = Pro_texts
        df['Con_text'] = Con_texes

    return

def update_ngrams(df_train: pd.DataFrame, df_test: pd.DataFrame):
    
    if 'Pro_ngram' in df_train.columns:
        print("ngrams features are already in dataframes")
        return

    vectorizer = TfidfVectorizer(max_features = 25000, sublinear_tf=False, max_df=0.9, stop_words='english', ngram_range=(1,3))
    vectorizer.fit(df_train[['Pro_text','Con_text']].values.flatten())
    Train_Pro_tfidf = vectorizer.transform(df_train['Pro_text'].tolist())
    Train_Con_tfidf = vectorizer.transform(df_train['Con_text'].tolist())
    Test_Pro_tfidf = vectorizer.transform(df_test['Pro_text'].tolist())
    Test_Con_tfidf = vectorizer.transform(df_test['Con_text'].tolist())
    df_train['Pro_ngram'] = [Train_Pro_tfidf[i] for i in range(Train_Pro_tfidf.shape[0])]
    df_train['Con_ngram'] = [Train_Con_tfidf[i] for i in range(Train_Con_tfidf.shape[0])]
    df_test['Pro_ngram'] = [Test_Pro_tfidf[i] for i in range(Test_Pro_tfidf.shape[0])]
    df_test['Con_ngram'] = [Test_Con_tfidf[i] for i in range(Test_Con_tfidf.shape[0])]
    

    return

def update_lexicon(df_train: pd.DataFrame, df_test: pd.DataFrame, CL_csv = r'lexica\connotation_lexicon_a.0.1.csv', NVL_csv= r'lexica\NRC-VAD-Lexicon-Aug2018Release\NRC-VAD-Lexicon.txt' ):

    df_CL = pd.read_csv(CL_csv, sep="_|,", names=['word', 'part', 'sentiment'])
    df_CL = df_CL.drop([54486, 54487])
    df_NVL = pd.read_csv(NVL_csv, sep="\t", names=['word', 'a-score', 'd-score', 'v-score'])

    if 'Pro_positive' in df_train.columns:
        print("connotation_lexicon features are already in dataframes")
        return
    
    positive_vectorizer = CountVectorizer(max_df=1, stop_words='english', vocabulary=set(df_CL[df_CL['sentiment'] == 'positive']['word'].values.astype('U')))
    neutral_vectorizer = CountVectorizer(max_df=1, stop_words='english', vocabulary=set(df_CL[df_CL['sentiment'] == 'neutral']['word'].values.astype('U')))
    negative_vectorizer = CountVectorizer(max_df=1, stop_words='english', vocabulary=set(df_CL[df_CL['sentiment'] == 'negative']['word'].values.astype('U')))

    for df in [df_train, df_test]:
        Pro_positive_count = positive_vectorizer.transform(df['Pro_text'])
        Con_positive_count = positive_vectorizer.transform(df['Con_text'])
        df['Pro_positive'] = [c.nnz for c in Pro_positive_count]
        df['Con_positive'] = [c.nnz for c in Con_positive_count]

        Pro_neutral_count = neutral_vectorizer.transform(df['Pro_text'])
        Con_neutral_count = neutral_vectorizer.transform(df['Con_text'])
        df['Pro_neutral'] = [c.nnz for c in Pro_neutral_count]
        df['Con_neutral'] = [c.nnz for c in Con_neutral_count]

        Pro_negative_count = negative_vectorizer.transform(df['Pro_text'])
        Con_negative_count = negative_vectorizer.transform(df['Con_text'])
        df['Pro_negative'] = [c.nnz for c in Pro_negative_count]
        df['Con_negative'] = [c.nnz for c in Con_negative_count]

    
    NVL_vectorizer = CountVectorizer(max_df=1, stop_words='english', vocabulary=df_NVL['word'].to_list())

    for df in [df_train, df_test]:
        NVL_score = coo_matrix(df_NVL[['a-score', 'd-score', 'v-score']].to_numpy())
        Pro_lexicon_count = NVL_vectorizer.transform(df['Pro_text'])
        Con_lexicon_count = NVL_vectorizer.transform(df['Con_text'])
        df[['Pro_a-score', 'Pro_d-score', 'Pro_v-score']] = pd.DataFrame(Pro_lexicon_count.dot(NVL_score).A, index=df.index)
        df[['Con_a-score', 'Con_d-score', 'Con_v-score']] = pd.DataFrame(Con_lexicon_count.dot(NVL_score).A, index=df.index)

    return

def upadate_linguistic(df_train: pd.DataFrame, df_test: pd.DataFrame):

    #length
    for df in [df_train, df_test]:
        df['Pro_Length'] = [len(text) for text in df['Pro_text']]
        df['Con_Length'] = [len(text) for text in df['Con_text']]

    #Modal verbs
    for df in [df_train, df_test]:
        modals = ['can', 'could', 'may', 'might', 'shall', 'should', 'will', 'would', 'must']
        modals_vectorizer = CountVectorizer(vocabulary=modals)
        Pro_Modals_count = modals_vectorizer.transform(df['Pro_text'])
        Con_Modals_count = modals_vectorizer.transform(df['Con_text'])
        df['Pro_Modals'] = [c.nnz for c in Pro_Modals_count]
        df['Con_Modals'] = [c.nnz for c in Con_Modals_count]

    #Questions
    for df in [df_train, df_test]:
        df['Pro_Questions'] = [ text.count('?')  for text in df['Pro_text'].tolist()]
        df['Con_Questions'] = [ text.count('?')  for text in df['Con_text'].tolist()]

    #Links to outside websites
    for df in [df_train, df_test]:
        df['Pro_Links'] = [ text.count('http')  for text in df['Pro_text'].tolist()]
        df['Con_Links'] = [ text.count('http')  for text in df['Con_text'].tolist()]
 
def update_user(df_train: pd.DataFrame, df_test: pd.DataFrame, user_loc = r'data\users.json'):

    '''
    big_issues_dict
    birthday
    education
    ethnicity
    gender
    friends
    income
    joined
    opinion_arguments
    opinion_questions
    party
    political_ideology
    poll_topics
    poll_votes
    relationship
    religious_ideology
    '''
    with open(r'data\users.json',  encoding='utf-8') as f:
        users = json.load(f)
    
    #Gender
    Gender_vectorizer = CountVectorizer(max_df=1, stop_words='english', vocabulary=set([p["gender"] for p in users.values()]))
    for df in [df_train, df_test]:
        Pro_gender = Gender_vectorizer.transform([users[person]['gender'] for person in df["pro_debater"].tolist()])
        Con_gender = Gender_vectorizer.transform([users[person]['gender'] for person in df["con_debater"].tolist()])
        df['Pro_gender'] = [Pro_gender[i] for i in range(Pro_gender.shape[0])]
        df['Con_gender'] = [Con_gender[i] for i in range(Con_gender.shape[0])]

    #Religious ideology
    RL_vectorizer = CountVectorizer(max_df=1, stop_words='english', vocabulary=set([p["religious_ideology"] for p in users.values()]))
    for df in [df_train, df_test]:
        Pro_RL = RL_vectorizer.transform([users[person]['religious_ideology'] for person in df["pro_debater"].tolist()])
        Con_RL = RL_vectorizer.transform([users[person]['religious_ideology'] for person in df["con_debater"].tolist()])
        df['Pro_religious_ideology'] = [Pro_RL[i] for i in range(Pro_RL.shape[0])]
        df['Con_religious_ideology'] = [Con_RL[i] for i in range(Con_RL.shape[0])]


    pass

def get_features(df_train: pd.DataFrame, df_test: pd.DataFrame, model = "ngrams"):
    
    # Initialize 'empty' features matrices
    x_train, x_test = coo_matrix(np.empty((df_train.shape[0], 0))), coo_matrix(np.empty((df_test.shape[0], 0)))

    # Get ngrams features if wanted
    if "Ngram" in model:

        x_train_Pro_ngrams = vstack(df_train['Pro_ngram'])
        x_train_Con_ngrams = vstack(df_train['Con_ngram'])
        x_test_Pro_ngrams = vstack(df_test['Pro_ngram'])
        x_test_Con_ngrams = vstack(df_test['Con_ngram'])
        x_train = hstack([x_train, x_train_Pro_ngrams, x_train_Con_ngrams])
        x_test = hstack([x_test, x_test_Pro_ngrams, x_test_Con_ngrams])


    if "Lex" in model:

        # Connotation Lexicon
        CL_features = ['Pro_negative', 'Con_positive', 'Pro_neutral', 'Con_neutral', 'Pro_negative', 'Con_negative']
        x_train = hstack([x_train, df_train[CL_features].values])
        x_test = hstack([x_test, df_test[CL_features].values])
        # NRC-VAD Lexicon
        NVL_features = ['Pro_a-score', 'Pro_d-score', 'Pro_v-score','Con_a-score', 'Con_d-score', 'Con_v-score']
        x_train = hstack([x_train, df_train[NVL_features].values])
        x_test = hstack([x_test, df_test[NVL_features].values])


    if "Ling" in model:
        Length_features = ['Pro_Length', 'Con_Length']
        Modals_features = ['Pro_Modals', 'Con_Modals', ]
        Questions_features = ['Pro_Questions', 'Con_Questions']
        Links_features = ['Pro_Links', 'Con_Links']
        x_train = hstack([x_train, df_train[Length_features+Modals_features+Questions_features+Links_features].values])
        x_test = hstack([x_test, df_test[Length_features+Modals_features+Questions_features+Links_features].values])
        

    if "User" in model:

        #Gender
        x_train_Pro_gender = vstack(df_train['Pro_gender'])
        x_train_Con_gender = vstack(df_train['Con_gender'])
        x_test_Pro_gender = vstack(df_test['Pro_gender'])
        x_test_Con_gender = vstack(df_test['Con_gender'])
        x_train = hstack([x_train, x_train_Pro_gender, x_train_Con_gender])
        x_test = hstack([x_test, x_test_Pro_gender, x_test_Con_gender])

        # religious_ideology
        x_train_Pro_RL = vstack(df_train['Pro_religious_ideology'])
        x_train_Con_RL = vstack(df_train['Con_religious_ideology'])
        x_test_Pro_RL = vstack(df_test['Pro_religious_ideology'])
        x_test_Con_RL = vstack(df_test['Con_religious_ideology'])
        x_train = hstack([x_train, x_train_Pro_RL, x_train_Con_RL])
        x_test = hstack([x_test, x_test_Pro_RL, x_test_Con_RL])
        

    return x_train, x_test

def get_lable(df_train: pd.DataFrame, df_test: pd.DataFrame):

    y_train = df_train['winner'].tolist()
    y_train = [1 if w=='Pro' else 0 for w in y_train]
    y_test = df_test['winner'].tolist()
    y_test = [1 if w=='Pro' else 0 for w in y_test]

    return y_train, y_test




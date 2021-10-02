import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy as sp
from scipy.sparse import coo_matrix, hstack


def get_dataframe(train_loc: str, test_loc: str):
    
    df_train = pd.read_json(train_loc,lines=True)
    df_test = pd.read_json(test_loc,lines=True)
    return df_train, df_test


def extract_text(df: pd.DataFrame) -> pd.DataFrame:

    df['winner'].replace('Pro', 1, inplace=True) 
    df['winner'].replace('Con', 0, inplace=True) 

    debate_log = []
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
        debate_log.append({'Pro_text':Pro_text, 'Con_text':Con_text})  
    
    
    df = pd.concat([df, pd.DataFrame(debate_log)], axis=1)
    
    return df


def get_ngrams(df_train: pd.DataFrame, df_test: pd.DataFrame):
    
    if 'Pro_text_tfidf' not in df_train.columns:
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.9, stop_words='english', ngram_range=(1,3))
        Pro_Con_corpus = df_train['Pro_text'].tolist() + df_train['Con_text'].tolist()
        vectorizer.fit(Pro_Con_corpus)
        df_train['Pro_text_tfidf'] = [ feature for feature in vectorizer.transform(df_train['Pro_text'].tolist())]
        df_train['Con_text_tfidf'] = [ feature for feature in vectorizer.transform(df_train['Con_text'].tolist())]
        df_test['Pro_text_tfidf'] = [ feature for feature in vectorizer.transform(df_test['Pro_text'].tolist())]
        df_test['Con_text_tfidf'] = [ feature for feature in vectorizer.transform(df_test['Con_text'].tolist())]

    return [df_train['Pro_text_tfidf'].tolist(), df_train['Con_text_tfidf'].tolist()], [df_test['Pro_text_tfidf'].tolist(), df_test['Con_text_tfidf'].tolist()]


def get_ngrams_simple(df_train: pd.DataFrame, df_test: pd.DataFrame):
    
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.9, stop_words='english', ngram_range=(1,3))
    Pro_Con_corpus = df_train['Pro_text'].tolist() + df_train['Con_text'].tolist()
    vectorizer.fit(Pro_Con_corpus)
    Train_Pro_tfidf = vectorizer.transform(df_train['Pro_text'].tolist())
    Train_Con_tfidf = vectorizer.transform(df_train['Con_text'].tolist())
    Test_Pro_tfidf = vectorizer.transform(df_test['Pro_text'].tolist())
    Test_Con_tfidf = vectorizer.transform(df_test['Con_text'].tolist())

    return [Train_Pro_tfidf, Train_Con_tfidf], [Test_Pro_tfidf, Test_Con_tfidf]


def get_features(df_train: pd.DataFrame, df_test: pd.DataFrame, model = "ngrams"):

    # Initialize 'empty' features matrices
    x_train_ngrams, x_test_ngrams = get_ngrams_simple(df_train, df_test)

    # Get ngrams features if wanted
    if "ngrams" in model:
        X_train_ngrams, X_test_ngrams = get_ngrams(df_train, df_test)
        # NEED TO hstack with scipy because you're dealing with SPARSE matrices
        X_train = hstack(x_train_ngrams)
        X_test = hstack(x_test_ngrams)

    return X_train, X_test


def get_lable(df_train: pd.DataFrame, df_test: pd.DataFrame):
    y_train = df_train['winner'].tolist()
    y_test = df_test['winner'].tolist()
    return y_train, y_test




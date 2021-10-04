import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
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


def upadate_ngrams(df_train: pd.DataFrame, df_test: pd.DataFrame):
    
    if 'Pro_text_tfidf' in df_train.columns:
        print("ngrams features are already in dataframes")
        return

    vectorizer = CountVectorizer(max_df=0.9, stop_words='english', ngram_range=(1,3) , max_features=None)
    Pro_Con_corpus = np.concatenate([df_train['Pro_text'].values,df_train['Con_text'].values])
    vectorizer.fit_ (Pro_Con_corpus)
    df_train['Pro_text_tfidf'] = [ feature for feature in vectorizer.transform(df_train['Pro_text'].tolist())]
    df_train['Con_text_tfidf'] = [ feature for feature in vectorizer.transform(df_train['Con_text'].tolist())]
    df_test['Pro_text_tfidf'] = [ feature for feature in vectorizer.transform(df_test['Pro_text'].tolist())]
    df_test['Con_text_tfidf'] = [ feature for feature in vectorizer.transform(df_test['Con_text'].tolist())]

    return


def get_ngrams_simple(df_train: pd.DataFrame, df_test: pd.DataFrame):
    
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.9, stop_words='english', ngram_range=(1,3))
    Pro_Con_corpus = df_train['Pro_text'].tolist() + df_train['Con_text'].tolist()
    vectorizer.fit(Pro_Con_corpus)
    Train_Pro_tfidf = vectorizer.transform(df_train['Pro_text'].tolist())
    Train_Con_tfidf = vectorizer.transform(df_train['Con_text'].tolist())
    Test_Pro_tfidf = vectorizer.transform(df_test['Pro_text'].tolist())
    Test_Con_tfidf = vectorizer.transform(df_test['Con_text'].tolist())

    return [Train_Pro_tfidf, Train_Con_tfidf], [Test_Pro_tfidf, Test_Con_tfidf]


def update_lexicon(df_train: pd.DataFrame, df_test: pd.DataFrame):

    if 'Pro_positive' in df_train.columns:
        print("connotation_lexicon features are already in dataframes")
        return

    df_conLex = get_lexicon_df(r'lexica\connotation_lexicon_a.0.1.csv')

    for df in [df_train, df_test]:

        positive_vectorizer = CountVectorizer(max_df=1, stop_words='english', vocabulary=set(df_conLex[df_conLex['sentiment'] == 'positive']['word'].values.astype('U')))
        Pro_positive_count = positive_vectorizer.transform(df['Pro_text'])
        Con_positive_count = positive_vectorizer.transform(df['Con_text'])
        df['Pro_positive'] = [c.nnz for c in Pro_positive_count]
        df['Con_positive'] = [c.nnz for c in Con_positive_count]

        neutral_vectorizer = CountVectorizer(max_df=1, stop_words='english', vocabulary=set(df_conLex[df_conLex['sentiment'] == 'neutral']['word'].values.astype('U')))
        Pro_neutral_count = neutral_vectorizer.transform(df['Pro_text'])
        Con_neutral_count = neutral_vectorizer.transform(df['Con_text'])
        df['Pro_neutral'] = [c.nnz for c in Pro_neutral_count]
        df['Con_neutral'] = [c.nnz for c in Con_neutral_count]

        negative_vectorizer = CountVectorizer(max_df=1, stop_words='english', vocabulary=set(df_conLex[df_conLex['sentiment'] == 'negative']['word'].values.astype('U')))
        Pro_negative_count = negative_vectorizer.transform(df['Pro_text'])
        Con_negative_count = negative_vectorizer.transform(df['Con_text'])
        df['Pro_negative'] = [c.nnz for c in Pro_negative_count]
        df['Con_negative'] = [c.nnz for c in Con_negative_count]

    return

def get_lexicon_df(connotation_csv = r'lexica\connotation_lexicon_a.0.1.csv'):
        connotation = pd.read_csv(connotation_csv, sep="_|,", names=['word', 'part', 'sentiment'])
        connotation = connotation.drop([54486, 54487])
        return connotation


def get_features(df_train: pd.DataFrame, df_test: pd.DataFrame, model = "ngrams"):

    # Initialize 'empty' features matrices
    x_train, x_test = coo_matrix(np.empty((df_train.shape[0], 0))), coo_matrix(np.empty((df_test.shape[0], 0)))

    # Get ngrams features if wanted
    if "ngrams" in model:
        # X_train_ngrams, X_test_ngrams = get_ngrams_simple(df_train, df_test)
        x_train_ngrams, x_test_ngrams = get_ngrams_simple(df_train, df_test)
        x_train = hstack([x_train] +  x_train_ngrams)
        x_test = hstack([x_test] + x_test_ngrams)


    if "lexicon" in model:
        update_lexicon(df_train, df_test)
        lexicon_feature_list = ['Pro_negative', 'Con_positive', 'Pro_neutral', 'Con_neutral', 'Pro_negative', 'Con_negative']
        x_train = hstack([x_train, df_train[lexicon_feature_list].values], )
        x_test = hstack([x_test, df_test[lexicon_feature_list].values], )

    

    return x_train, x_test


def get_lable(df_train: pd.DataFrame, df_test: pd.DataFrame):
    y_train = df_train['winner'].tolist()
    y_test = df_test['winner'].tolist()
    return y_train, y_test




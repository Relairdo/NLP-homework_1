import argparse
import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from scipy.sparse.construct import vstack
from features import get_dataframe, update_text, update_ngrams, update_lexicon, upadate_linguistic, update_user, get_features, get_lable
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, plot_confusion_matrix
from sklearn.model_selection import cross_val_score


parser = argparse.ArgumentParser()
parser.add_argument('--train', dest='train', required=False, default='data/train.jsonl',
                    help='Full path to the training file')
parser.add_argument('--test', dest='test', required=False, default='data/val.jsonl',
                    help='Full path to the evaluation file')
parser.add_argument('--user_data', dest='user_data', required=False, default='data/users.json',
                    help='Full path to the user data file')
parser.add_argument('--model', dest='model', required=False, default='Ngram+Lex+Ling+User',
                    choices=["Ngram", "Ngram+Lex", "Ngram+Lex+Ling", "Ngram+Lex+Ling+User"],
                    help='The name of the model to train and evaluate.')
parser.add_argument('--lexicon_path', dest='lexicon_path', required=False, default='lexica/',
                    help='The full path to the directory containing the lexica.'
                            ' The last folder of this path should be "lexica".')
parser.add_argument('--outfile', dest='outfile', required=False, default='out.txt',
                    help='Full path to the file we will write the model predictions')
                    
args = parser.parse_args("")

folderpath = os.path.join('pickle')
if os.path.isdir(folderpath):
    pass
else:
    os.mkdir(folderpath)

df_train_loc = os.path.join(folderpath, 'df_train.pkl')
df_test_loc = os.path.join(folderpath, 'df_test.pkl')
df_user_loc = os.path.join(folderpath, 'df_user.pkl')


if os.path.isfile(df_train_loc) and os.path.isfile(df_test_loc):
    df_train = pd.read_pickle(df_train_loc)
    df_test = pd.read_pickle(df_test_loc)
    df_user = pd.read_pickle(df_user_loc)

else:

    start = time.time()
    
    df_train, df_test = get_dataframe(args.train, args.test)
    update_text(df_train, df_test)
    update_ngrams(df_train, df_test, feature_number=1000)
    update_lexicon(df_train, df_test, args.lexicon_path)
    upadate_linguistic(df_train, df_test)
    df_train, df_test, df_user = update_user(df_train, df_test, args.user_data)

    end = time.time()
    print("Data Preprocessiong Cost:", round(end - start),'s.')

    df_train.to_pickle(df_train_loc)
    df_test.to_pickle(df_test_loc)
    df_user.to_pickle(df_user_loc)

import itertools
def get_all_combinations(l : list, choose2=False) -> list:
    ll = []
    for L in range(0, len(l)+1):
        for subset in itertools.combinations(l, L):
            if choose2:
                if len(list(subset)) == 2:
                    ll.append(list(subset))
            else:
                ll.append(list(subset))
    return ll

lexicons_list = ["CL", "NVL"]
ling_feature_list = ['Length', 'R2O', 'Personal_pronouns', 'Modals', 'Links', 'Questions']
user_feature_list = ['education','ethnicity', 'gender', 'income', 'joined', 'party', 'political_ideology', 'relationship', 'religious_ideology']
all_feature_list = list(lexicons_list+ling_feature_list+user_feature_list)

df_train_r = df_train[df_train['category']=='Religion']
df_train_nr = df_train[df_train['category']!='Religion']
df_test_r = df_test[df_test['category']=='Religion']
df_test_nr = df_test[df_test['category']!='Religion']

column_names = ["Religion", "Lex","Ling","User","5FCV Mean"]
df_record = pd.DataFrame(columns = column_names)
for lex in lexicons_list:
    for ling in get_all_combinations(ling_feature_list, choose2=True):
        for user in get_all_combinations(user_feature_list, choose2=True):
            for religion in [True, False]:
                if religion:
                    x_train, x_test = get_features(df_train_r, df_test_r, df_user, model = args.model,lex_list=lex, ling_list=ling, user_list=user)
                    y_train, y_test = get_lable(df_train_r, df_test_r)
                else:
                    x_train, x_test = get_features(df_train_nr, df_test_nr, df_user,model = args.model,lex_list=lex, ling_list=ling, user_list=user)
                    y_train, y_test = get_lable(df_train_nr, df_test_nr)
                x = vstack([x_train,x_test])
                y = y_train + y_test
                clf = LogisticRegression(solver='liblinear', max_iter=500)
                scores = cross_val_score(clf, x, y, cv=5 ,scoring='accuracy')
                mean_score = np.mean(scores)
                record = {"Religion":religion, "Lex":lex,"Ling":ling,"User":user,"5FCV Mean":mean_score}
                df_record = df_record.append(record,ignore_index=True)

            df_record.to_csv(os.path.join('Traversal_religion.csv'))

print("Finished")
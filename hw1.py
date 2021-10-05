# %%
import argparse
import pandas as pd
from features import get_dataframe, update_text, update_ngrams, update_lexicon, upadate_linguistic, update_user, get_features, get_lable
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, plot_confusion_matrix
import time
# %%
if __name__ == '__main__':
    pass
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--train', dest='train', required=True,
    #                     help='Full path to the training file')
    # parser.add_argument('--test', dest='test', required=True,
    #                     help='Full path to the evaluation file')
    # parser.add_argument('--user_data', dest='user_data', required=True,
    #                     help='Full path to the user data file')
    # parser.add_argument('--model', dest='model', required=True,
    #                     choices=["Ngram", "Ngram+Lex", "Ngram+Lex+Ling", "Ngram+Lex+Ling+User"],
    #                     help='The name of the model to train and evaluate.')
    # parser.add_argument('--lexicon_path', dest='lexicon_path', required=True,
    #                     help='The full path to the directory containing the lexica.'
    #                          ' The last folder of this path should be "lexica".')
    # parser.add_argument('--outfile', dest='outfile', required=True,
    #                     help='Full path to the file we will write the model predictions')
    # args = parser.parse_args()


    # Homework Start From Here
    
# %%|
df_train = pd.read_pickle(r'df_train.pkl')
df_test = pd.read_pickle(r'df_test.pkl')


# %%
start = time.time()

df_train, df_test = get_dataframe(r'data\train.jsonl', r'data\val.jsonl')
update_text(df_train, df_test)
update_ngrams(df_train, df_test)
update_lexicon(df_train, df_test)
upadate_linguistic(df_train, df_test)
update_user(df_train, df_test)

end = time.time()
print("Data Preprocessiong Cost:", round(end - start),'s.')
# %%
df_train.to_pickle('df_train.pkl')
df_test.to_pickle('df_test.pkl')

# %% 
x_train, x_test = get_features(df_train, df_test, model = "Ngram+Lex+Ling+User")
y_train, y_test = get_lable(df_train, df_test)
print('total features:', x_train.shape[1])
# %% Swap
# x_train, x_test = x_test, x_train
# y_train, y_test = y_test, y_train
# %%
start = time.time()
clf = LogisticRegression()
clf.fit(x_train, y_train)
end = time.time()
print("Training Model Cost:", round(end - start),'s.')
# %%
y_predicted_LR = clf.predict(x_test)
print('ngram-LR Classification:')
print("Accuracy score: ",accuracy_score(y_test, y_predicted_LR))
print("Accuracy score on Train: ",accuracy_score(y_train, clf.predict(x_train)))
plot_confusion_matrix(clf, x_test, y_test)



# %%
del clf
# %%

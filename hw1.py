# %%
import argparse
from features import get_dataframe, extract_text, get_features, get_lable
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, plot_confusion_matrix

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
    
# %%
df_train, df_test = get_dataframe(r'data\train.jsonl', r'data\val.jsonl')
df_train, df_test = extract_text(df_train), extract_text(df_test)
x_train, x_test = get_features(df_train, df_test, model = "ngrams+lexicon")
y_train, y_test = get_lable(df_train, df_test)


# %% 
# x_train, x_test = x_test, x_train
# y_train, y_test = y_test, y_train
# %%
clf = LogisticRegression()
clf.fit(x_train, y_train)
y_predicted_LR = clf.predict(x_test)

print('ngram-LR Classification:')
print("Accuracy score: ",accuracy_score(y_test, y_predicted_LR))
plot_confusion_matrix(clf, x_test, y_test)

# %%

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

if __name__ == '__main__':
    
    '''
        hw1.py --train <path_to_resources>/data/train.jsonl 
        --test <path_to_resources>/data/dev.jsonl 
        --model "Ngram+Lex" 
        --lexicon_path <path_to_resources>/lexica/
        --user_data <path_to_resources>/data/users.json
        --outfile <path_to_output_file>
    '''

    # python hw1.py --train data/train.jsonl --test data/val.jsonl --model "Ngram+Lex+Ling+User" --lexicon_path lexica/ --user_data data/users.json --outfile out.txt

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', dest='train', required=True,
                        help='Full path to the training file')
    parser.add_argument('--test', dest='test', required=True,
                        help='Full path to the evaluation file')
    parser.add_argument('--user_data', dest='user_data', required=True,
                        help='Full path to the user data file')
    parser.add_argument('--model', dest='model', required=True,
                        choices=["Ngram", "Ngram+Lex", "Ngram+Lex+Ling", "Ngram+Lex+Ling+User"],
                        help='The name of the model to train and evaluate.')
    parser.add_argument('--lexicon_path', dest='lexicon_path', required=True,
                        help='The full path to the directory containing the lexica.'
                             ' The last folder of this path should be "lexica".')
    parser.add_argument('--outfile', dest='outfile', required=True,
                        help='Full path to the file we will write the model predictions')
    args = parser.parse_args()

    cross_validation = False

    # Homework Start From Here
        
    df_train_loc = os.path.join('df_train.pkl')
    df_test_loc = os.path.join('df_test.pkl')
    df_user_loc = os.path.join('df_user.pkl')


    if os.path.isfile(df_train_loc) and os.path.isfile(df_test_loc) and os.path.isfile(df_user_loc):
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


    x_train, x_test = get_features(df_train, df_test, df_user, model = args.model)
    y_train, y_test = get_lable(df_train, df_test)
    print('total features:', x_train.shape[1])

    if cross_validation:
        x = vstack([x_train,x_test])
        y = y_train + y_test
        print(args.model)
        clf = LogisticRegression(solver='liblinear')
        start = time.time()
        scores = cross_val_score(clf, x, y, cv=5 ,scoring='accuracy')
        end = time.time()
        print("Training Model Cost:", round(end - start),'s.')
        print(scores)
        print(np.mean(scores))

    else:

        start = time.time()
        clf = LogisticRegression(solver='liblinear')
        clf.fit(x_train, y_train)
        end = time.time()
        print("Training Model Cost:", round(end - start),'s.')
        
        y_predicted = clf.predict(x_test)
        print('ngram-LR Classification:')
        print("Accuracy score: ",accuracy_score(y_test, y_predicted))
        print("Accuracy score on Train: ",accuracy_score(y_train, clf.predict(x_train)))
        plot_confusion_matrix(clf, x_test, y_test)
        print(classification_report(y_test, y_predicted, target_names=['Con','Pro']))

        with open(args.outfile, "w") as myfile:
            for p in y_predicted:
                if p == 1:
                    myfile.write('Pro'+'\n')
                else:
                    myfile.write('Con'+'\n')
        
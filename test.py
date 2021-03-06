import argparse
if __name__ == '__main__':

    '''
    hw1.py --train <path_to_resources>/data/train.jsonl 
    --test <path_to_resources>/data/dev.jsonl 
    --model "Ngram+Lex" 
    --lexicon_path <path_to_resources>/lexica/
    --user_data <path_to_resources>/data/users.json
    --outfile <path_to_output_file>
    '''

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
    print(args.train)
    # Homework Start From Here
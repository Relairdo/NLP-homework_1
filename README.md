# Homework 1: Supervised Text Classification
 #### 1 late day
 #### Kuan-Hsuan Wu (Frank)
 #### Homewrok number 1
 
 ## Train and Test
 python hw1.py --train <path_to_resources>/data/train.jsonl --test <path_to_resources>/data/dev.jsonl --model "Ngram+Lex" --lexicon_path <path_to_resources>/lexica/ --user_data <path_to_resources>/data/users.json --outfile <path_to_output_file>
 
 The program will read data fron josnl or json and load it into dataframe.
 Than upadate each features into datafram.
 After upadted all the features in datafram, get_feature can extract feature from the dataframes.
 
 It will create a piclke/ folder and place 3 pickle file of three dataframe df_train, df_test, df_user in side. In the first time Train and Test.
 
 I use another file analyize.ipynb to analyze and do expiriments, but the submission only can include hw1.py and feature.py and a README.md, details is in writing part.
 
 
 ## Feature and Classifirer
Following are links to my expiriment result correspond to my written part.
 https://docs.google.com/spreadsheets/d/1hCsXtP3Ex-nli0ISjOp71lYu5tqlXQfsxi_myOQwrao/edit?usp=sharing
 https://docs.google.com/spreadsheets/d/1zPp1s3sMah6hvfLcGyQ0u1Ht0OiaCXeTlugbJvRU4sk/edit?usp=sharing
 https://docs.google.com/spreadsheets/d/1JyWagFWWO2hcslsMxQxdK9CLAb8QtmRlhIj_56e9YY8/edit?usp=sharing 
 https://docs.google.com/spreadsheets/d/1rja_Xs8ZJ-EPqUhgmmShiAm8_As9W8Z4bvTKYHiXi1A/edit?usp=sharing
 

#### Ling Featurs I implements (* mean what I choose)
1. Length 
2. Reference to the opponent
3. Personal pronouns
4. Modal verbs 
5. Links to outside websites *
6. Questions *
    
#### User Features I implements (* mean what I choose)
1. Education *
2. Ethnicity
3. Gender
4. Income
5. Joined
6. Party *
7. Political Ideology
8. Relationship
9. Religious Ideology

to change the feature please modify the args of get_features  
get_features(df_train, df_test, df_user, model = args.model, lex_list=["NVL"], ling_list=['Links', 'Questions'], user_list=['education', 'party'])  
 
to use all result just change to get_features(df_train, df_test, df_user, model = args.model)  
it have better result than only use 2 Ling and 2 User  

More detail in writing part...  

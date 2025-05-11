from hazdet.train import hyperparameter_tune_all_models
# If you have raw qualtrics data, we have helper functions to clean it
# e.g., X,y =  create_features(file='Tweet annotation.csv')

# saves best model as 'finalized_model_'+best_model+'.sav'
filename ='../ground_truth_data/TweetGT.csv'
hyperparameter_tune_all_models(filename)

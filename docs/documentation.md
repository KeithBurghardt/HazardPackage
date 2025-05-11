

# Inference details

The basic way to extract predictions is via `inference()`:
```
predictions = inference(filename,model_filename,text_col='text',sentence_tf= 'stsb-xlm-r-multilingual')
```
## Parameters
- **filename**: file name or path to file name
  The filename should end in the following: { '.csv' (CSV file),'.tsv' (TSV file),'.json' (JSON file),'.xml' (XML file), '.xls' or 'xlsx' (Excel file), '.hdf' (HDF file), '.sql' (SQL data).
- **model_filename**: path to the model file
  The current file is located in hazdet/finalized_model_SVM.sav, but this can be modified with a new model. We assume this can be opened with pickle.
- **text_col**: text column name (_default: 'text'_)
- **sentence_tf**: name of sentence transformer that create post representations. 
The name should be anything that can be run via SentenceTransformer(sentence_tf). Up-to-date pre-trained models are listed here: https://www.sbert.net/docs/sentence_transformer/pretrained_models.html.

## Output
Pandas dataframe
- Includes all columns of the original dataframe
- Adds additional column called "hazard", which is the hazard confidence.


## Model file
hazdet/finalized_model_SVM.sav: the SVM model that detects hazards from XLM-RoBERTa embeddings


# Training details
To train, you run the following
```
from hazdet.train import hyperparameter_tune_all_models
# saves best model as 'finalized_model_'+best_model+'.sav'
filename ='../ground_truth_data/TweetGT.csv'
hyperparameter_tune_all_models(filename)
```
In the background, this embeds all text data into a sentence transformer (default: 'stsb-xlm-r-multilingual'), then trains a Random Forest classifier, Support Vector Classifier, XGBoost classifier, and basic feed-forward neural network on these data (with hyperparameters optimized) and finds which model of all of them is the best (based on ROC-AUC). The best model is saved in a pickle file. 

# Data
## Data collection
- data/extract_tweets_threats.py: extract treat posts from the X API (code is ca. 2022) - posts are collected across entire X history
- data/extract_tweets_commonwords.py: extract common posts from the X API (code is ca. 2022) - posts are collected across entire X history
## Annotation guidelines
- data/Hazard_guidelines.docx: Annotator guidelines
## Dataset raw data and metadata
- data/datasheet-for-hazard-dataset.md: Datasheet for dataset "X Post Hazard Ground Truth Dataset"
- data/AnonymousAnnotations.csv: annotations for each post by each annotator. We include an anonymized annotator ID along with basic demographics to determine whether demographics affect what is annotated as a hazard. We also include severe hazards and benefit annotations, which we do not analyze in our study.
## Training data
- data/TweetGT.csv: the mean of annotator labels for each post based on AnonymousAnnotations.csv. We also include severe hazards and benefit annotations, which we do not analyze in our study.

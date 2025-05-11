# Data
All code to collect training data is in ground_truth_data/. 

## Data collection
- extract_tweets_threats.py: extract treat posts from the X API (code is ca. 2022) - posts are collected across entire X history
- extract_tweets_commonwords.py: extract common posts from the X API (code is ca. 2022) - posts are collected across entire X history
## Annotation guidelines
- Hazard_guidelines.docx: Annotator guidelines
## Dataset raw data and metadata
- datasheet-for-hazard-dataset.md: Datasheet for dataset "X Post Hazard Ground Truth Dataset"
- AnonymousAnnotations.csv: annotations for each post by each annotator. We include an anonymized annotator ID along with basic demographics to determine whether demographics affect what is annotated as a hazard. We also include severe hazards and benefit annotations, which we do not analyze in our study.
## Training data
- TweetGT.csv: the mean of annotator labels for each post based on AnonymousAnnotations.csv. We also include severe hazards and benefit annotations, which we do not analyze in our study.
--
# 
to install use pip install -e .   

- hazard_detection.py: code to extract hazards from X posts in JSONL format.
- finalized_model_SVM.sav: the SVM model that detects hazards from XLM-RoBERTa embeddings


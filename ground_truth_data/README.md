## Files:
- extract_tweets_threat.py: our code to collect tweets that contain words from the threat dictionary. This dictionary is here: - https://www.michelegelfand.com/threat-dictionary
- extract_tweets_common_words.py: our code to collect tweets that contain common English words based on the top lemmas from https://www.wordfrequency.info/samples.asp
- Hazard guidelines: text annotators use to detect if a hazard exists 
- AnonymousAnnotations.csv: annotations for each post by each annotator. We include an anonymized annotator ID along with basic demographics to determine whether demographics affect what is annotated as a hazard. We also include severe hazards and benefit annotations, which we do not analyze in our study.
- TweetGT.csv: the mean of annotator labels for each post based on AnonymousAnnotations.csv. We also include severe hazards and benefit annotations, which we do not analyze in our study.
- datasheet-for-hazard-dataset.md: The datasheet (from Datasheets for Datasets by Gebru et al., 2021) for the X posts shown in  AnonymousAnnotations.csv and TweetGT.csv.

## References:

Fessler, D. M., Pisor, A. C., & Navarrete, C. D. (2014). Negatively-biased credulity and the cultural evolution of beliefs. PloS one, 9(4), e95167.

Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Iii, H. D., & Crawford, K. (2021). Datasheets for datasets. Communications of the ACM, 64(12), 86-92.

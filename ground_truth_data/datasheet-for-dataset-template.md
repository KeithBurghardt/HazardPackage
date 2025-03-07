# Datasheet for dataset "X Post Hazard Ground Truth Dataset"

Questions from the [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) paper, v7.

Jump to section:

- [Motivation](#motivation)
- [Composition](#composition)
- [Collection process](#collection-process)
- [Preprocessing/cleaning/labeling](#preprocessingcleaninglabeling)
- [Uses](#uses)
- [Distribution](#distribution)
- [Maintenance](#maintenance)

## Motivation
We are motivated to annotate tweets by the existance of hazards because there is substantial literature suggesting hazards are a unique text indicator dimension to analyze in part because evidence suggests people are more likely to believe hazards than benefits.

### For what purpose was the dataset created? 

The purpose of these data collection is to extract a representative sample of X post from which to annotate hazards.


### Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?

This is anonymous to preserve the double-blind review.

### Who funded the creation of the dataset? 

This is anonymous to preserve the double-blind review.

### Any other comments?
N/A
## Composition

These were random posts collected via the X Academic API between 2006 and 2022. These were posts that contained words from the Thret Dictionary, including ``anguish'', ``conflict'', and ``recession''. 

### What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)?

These instances represent individual posts, either original posts or replies.

### How many instances are there in total (of each type, if appropriate)?

There are 1.3K posts that are annotated at all. After filtering, we have 1.1K posts for our analysis (we recommend removing posts that were not annotated by at least 2 annotators and that were completed under 200 seconds).

### Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set?

These data are a sample. It is acquired from the X API, which might not collect these data independently at random. Without access to the full X data repository, we cannot fully ascertain how representative these data are, but their use of English language threat words implies these are more likely representative of English language posts containing threat words.

### What data does each instance consist of? 

The unprocessed data are text, along with individual annotations indicating if the text are spam, if the text contains a hazard, if  a severe hazard, a benefit, or a severe benefit (it might contain multiple labels). We also ask annotators if they believe the author of the text believes what they wrote.

### Is there a label or target associated with each instance?

Yes, see AnonymousAnnotations.csv

### Is any information missing from individual instances?

We do not include text IDs, author IDs, or when the text was written.

### Are relationships between individual instances made explicit (e.g., users’ movie ratings, social network links)?

N/A

### Are there recommended data splits (e.g., training, development/validation, testing)?

N/A

### Are there any errors, sources of noise, or redundancies in the dataset?

No.
### Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)?

These data are self-contained.

### Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor-patient confidentiality, data that includes the content of individuals’ non-public communications)?

There aree no data that we believe are considered confidential.

### Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?

This data may contain offensive text, but we accept this potential harm as hazards necessarily may describe something offensive. By removing such text, we believe the model may not have very representative data.

### Does the dataset relate to people? 

These data describe the demographics of annotators, but is non-human subject research.

### Does the dataset identify any subpopulations (e.g., by age, gender)?

We annotate gender, age, political orientation, and whether the annotator is a parent (some research suggests parents view hazards differently than non-parents).

### Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset?

No.

### Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)?

No.

### Any other comments?
N/A
## Collection process

These data contain X original posts or replies with at least one word in the Threat Dictionary.

### How was the data associated with each instance acquired?

These are raw text from the X API.

### What mechanisms or procedures were used to collect the data (e.g., hardware apparatus or sensor, manual human curation, software program, software API)?

We access the X API via a local server.

### If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?

Sampling was stochastic.

### Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?

These data were collected by a researcher who was compensated via their salary.

### Over what timeframe was the data collected?

These data were collected in late 2022, and contain posts since 2006.

### Were any ethical review processes conducted (e.g., by an institutional review board)?

These data were non-human subject research.

### Does the dataset relate to people?

These data are publicly gathered and are therefore non-human subject research.

### Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources (e.g., websites)?

Posts are gathered via third parties (the X API).

### Were the individuals in question notified about the data collection?

Authors of collected posts were not notified.

### Did the individuals in question consent to the collection and use of their data?

These are public data collected from an API and anonymized. We therefore did not need consent in our country.

### If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses?

N/A

### Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis) been conducted?

N/A

### Any other comments?
N/A
## Preprocessing/cleaning/labeling

In TweetGT.csv, we change emojis to words in order to improve text embeddings. Otherwise, text is not modified.

### Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)?

No.

### Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)?

Yes, see TweetGT.csv's "old_text" column.

### Is the software used to preprocess/clean/label the instances available?

Yes, we use the Python demoji library (https://pypi.org/project/demoji/). 

### Any other comments?
N/A
## Uses

These data can be used to train a model that annotates hazards or benefits (or potentially spam) at scale. These data should not be used to find individuals who are sharing what a model detects as hazards as the model could be wrong. 

### Has the dataset been used for any tasks already?

Yes, these data have been used to train several hazard detection models.

### Is there a repository that links to any or all papers or systems that use the dataset?

This is anonymous to preserve the double-blind review.


### What (other) tasks could the dataset be used for?
N/A
### Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?

These data may not be representative of posts made since 2022. In addition, these data might not be adequate to train a model to detect hazards in other platforms/text because of data shift, in which the test set is too different from the training set.

### Are there tasks for which the dataset should not be used?

These data should not be used to find individuals who appear to share hazards.

### Any other comments?
N/A
## Distribution

### Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created? 

Yes, these data are freely available.

### How will the dataset will be distributed (e.g., tarball on website, API, GitHub)?

These data are available within thie GitHub folder.

### When will the dataset be distributed?
These data are available now.

### Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?

No.

### Have any third parties imposed IP-based or other restrictions on the data associated with the instances?

No.

### Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?

No.

### Any other comments?
N/A
## Maintenance



### Who is supporting/hosting/maintaining the dataset?
Metadata will be updated by the author.

### How can the owner/curator/manager of the dataset be contacted (e.g., email address)?

This is anonymous to preserve the double-blind review.

### Is there an erratum?

If there will be any errara, it will be listed in the README.md

### Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)?

These data is expected to be static.
### If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were individuals in question told that their data would be retained for a fixed period of time and then deleted)?

N/A
### Will older versions of the dataset continue to be supported/hosted/maintained?

Yes, if these data were to be changed, we will have older versions in a separate folder.
### If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?
Yes, other researchers can create a GitHub branch.

### Any other comments?
N/A

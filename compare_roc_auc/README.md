# Comparing ROC-AUC

Here we show code to compare ROC-AUC values.

Basic libraries

```
pip install pandas numpy seaborn matplotlib scikit-learn scipy statsmodels demoji
pip install -U sentence-transformers
pip install xgboost
pip install "tensorflow<2.16"
pip install tf-keras
pip install --force-reinstall --no-cache-dir "numpy<2" "ml-dtypes<0.3.2"
pip install scikit-optimize
pip install GPyOpt
```

To plot ROC-AUC values: 
- Run `hazard_roc_auc.py` to get ROC-AUC for various embedding models applied to SVM, Random Forest, XGBoost, and a feed-forward neural network (all hyperparameters optimized by Bayesian optimization).
- Then run `plot_data.py` to plot the mean and standard deciations of ROC-AUC for these models as well as 2 baselines: LLMs (GPT-3.5, 4, and 5 with zero-shot, two-shot, and five-shot prompts; cf. llm_annotation.py) and a dictionary method (threat dictionary plus 3 nearest synonyms for each word, based on Word2Vec).

Datasets (extracted from hazard_roc_auc.py):
- tweet_responses_2025.csv: LLM responses for each ground truth label with zero-shot prompting
- tweet_responses_2025_n=2/5.csv: LLM responses for each ground truth label with two- or five-shot prompting
- metrics_gpt_vs_gpt_s_new_Qwen_Qwen3-Embedding-0.6B_correct_split.csv: ROC-AUC for `Qwen3-Embedding-0.6B` embedding model (bootstrapped within a test dataset)
- metrics_gpt_vs_gpt_s_new_sentence-transformers_distiluse-base-multilingual-cased-v2_correct_split.csv: ROC-AUC for `distiluse-base-multilingual-cased-v2` embedding model (bootstrapped within a test dataset)
- metrics_gpt_vs_gpt_s_new_sentence-transformers_paraphrase-multilingual-MiniLM-L12-v2_correct_split.csv: ROC-AUC for `paraphrase-multilingual-MiniLM-L12-v2` embedding model (bootstrapped within a test dataset)
- metrics_gpt_vs_gpt_s_new_stsb-xlm-r-multilingual_correct_split.csv: ROC-AUC for `stsb-xlm-r-multilingual` embedding model (bootstrapped within a test dataset)
- hyperparameters.txt: the best hyperparameters for each model studied using `distiluse-base-multilingual-cased-v2`, the best-performing embedding model



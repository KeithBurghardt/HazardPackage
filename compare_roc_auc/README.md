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

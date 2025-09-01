import nltk
import gensim
import pandas as pd
from nltk.data import find 
print('making word2vec model')
nltk.download('word2vec_sample')
word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)


threat_words = pd.read_csv('threat.txt',header=None)
threat_words.columns = ['word']
print('finding top words')

for num_top in [2,5]:
    comprehensive_words = set()
    for word in threat_words.word.values:
        try:
            top=model.most_similar(positive=[word], topn = num_top)
            comprehensive_words.add(word.lower())
            for synonym,score in top:
                comprehensive_words.add(synonym.lower())
        except:
            continue
    comprehensive_words = pd.DataFrame({'word':list(comprehensive_words)})
    comprehensive_words.to_csv('comprehensive_words_n='+str(num_top)+'.csv',index=False)

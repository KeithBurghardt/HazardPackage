# How to install
To install, just use pip install -e .   

# Quick start
```
from hazdet.inference import inference
from glob import glob
filename = 'bin/text.csv'
model_filename = list(glob('*/*.sav'))[0]
# We assume that text is in a column "text"    
predictions = inference(file,model_filename)
```
**Output**: Pandas dataframe of file with additional prediction column labeled "hazard"
# More details
See docs/ for more details of how to predict hazards, train models, and information about data we store.

# How to train your own model
We offer raw data, and data collection code in ground_truth_data/. 




# AI model to detect hazard information in text
This is a command line interface (CLI) that determines hazard confidence of a given text.

![Hazard detection](images/AIHazardDetectionIcon.png | width=100)
## How to install
Requirements: Python 3.12, pip (e.g., pip-25.1.1 works)
- Download the repo via git clone, or you can download it and unzip it
- Create a new Python environment via venv (see below)
- Install required libraries via pip install -e .
```
git clone git@github.com:username/repo.git #(anonymized for review purposes)
cd repo #(anonymized for review purposes)
python -m venv newenv
source newenv/bin/activate
pip install -e . # installs all required libraries
```
## Quick start
```
from hazdet.inference import inference
filename = 'bin/text.csv'
model_filename = 'hazdet/finalized_model_SVM.sav'
# We assume that text is in a column "text"
text_col = 'text'
# predictions is a Pandas dataframe
predictions = inference(file,model_filename,text_col=text_col)
# if you want to label text as hazard or not, rather than output confidences:
predictions['hazard'] = predictions['hazard'].to_numpy().round()
# optional: save just the data, including hazard confidence
predictions.to_csv('predictions.csv',index=False)
```
**Output of inference()**: Pandas dataframe of file with additional prediction column labeled "hazard"

[Inference](bin/quickstart_inference.py) and [training](bin/quickstart_train.py) examples can be found in `bin/`

## More details 
See the [documentation](docs/documentation.md) for more details of how to predict hazards, train models, and information about data we store.

## How to train your own model
We offer raw data, and data collection code in `ground_truth_data/`. 

## Citation
Please cite our work when using it in a paper or document:
```
@inproceedings{AnonymousHazards2025,
Title={Posts of Peril: Detecting Hazards in Text},
Authors={Anonymized for review},
Year={2025},
Booktitle={Under review},
Pages={11}
}
```
Or
```
Anonymized. (2025). Posts of Peril: Detecting Hazards in Text. Under review.
```




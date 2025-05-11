from hazdet.inference import inference
from glob import glob
def main():
    # We assume that text is in a column "text"
    file = 'text.csv'
    model_file = list(glob('../*/*.sav'))[0]
    predictions = inference(file,model_file)
    

__init__():
    main()

from hazdet.inference import inference
from glob import glob
def main():
    # We assume that text is in a column "text"
    filename = 'text.csv'
    model_filename = list(glob('../*/*.sav'))[0]
    predictions = inference(filename,model_filename)
    

__init__():
    main()

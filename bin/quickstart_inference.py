from hazdet.inference import inference
from glob import glob
def main():
    # We assume that text is in a column "text"
    filename = 'text.csv'
    model_filename = list(glob('../*/*.sav'))[0]
    # Output: Pandas dataframe of file in text.csv with additional column called "hazard", 
    # which lists the hazard confidence for text in each row
    predictions = inference(filename,model_filename)
    

__init__():
    main()

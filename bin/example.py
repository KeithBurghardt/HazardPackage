from hazdet.inference import inference
from glob import glob
def main():
    file = 'test.csv'
    model_file = list(glob('../*/*.sav'))[0]
    predictions = inference(file,model_file)
    

__init__():
    main()

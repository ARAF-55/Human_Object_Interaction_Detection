##### This script will refine the predictions based on detected object by the object detector. Following by the work of https://github.com/vt-vl-lab/iCAN#######


import numpy as np
import pickle

# Open the pickle file in binary mode and specify the encoding
with open('infos/prior.pickle', 'rb') as fp:
    priors = pickle.load(fp, encoding='latin1')  # Add encoding='latin1'

def apply_prior(Object, prediction_HOI_in):
    prediction_HOI = np.ones(prediction_HOI_in.shape)
    
    for index, prediction in enumerate(prediction_HOI):
        prediction_HOI[index] = priors[int(Object[index])]
        
    return prediction_HOI

                            


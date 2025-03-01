
import matplotlib.pyplot as plt
import numpy as np
import mne
from scipy import stats
from scipy import integrate
from scipy import signal
from sklearn.decomposition import PCA


# emg feature extraction:
# work with raw, not preprocessed data for time series signals (but paper 2 says to use butterworth filter first?)
# construct mx37 matrix for 37 feature formulas from paper 1

# try eliminating frequency domain features except MNF and PSR, due to poor perfomance in classification
# also could eliminate the features from prediction model and time dependence method (perform poor on classfication)


def compute_features(segment):
    features = []
    
    # four categories of similar time-domain features:

    # 1. energy and complexity
    iemg = np.sum(np.abs(segment), axis=1)
    features.extend(iemg)

    mav = np.mean(np.abs(segment), axis=1)
    features.extend(mav)

    #****REDO REST WRONG SHAPE****

    # MAV2
    # SSI
    #VAR
    # RMS
    # V
    # LOG

    # WL, AAC, and DASD
    
    # 2. frequency information
    # ZC, MYOP, WAMP, SSC

    # 3. prediction model method
    # AR, CC

    # 4. time dependence method
    #  MAVS, MHW, and MTW

    # frequency domain
    #MNP, TTP, SM1, SM2,SM3

    return np.array(features)

def extract_features(data_segments):
    return np.array([compute_features(segment) for segment in data_segments])



# testing dim with randomly generated data del later
if __name__ == "__main__":
    # Simulated data
    num_segments = 80
    numchannels = 2
    segment_length = 500  # samples per segment
    
    data_segments = [np.random.randn(numchannels, segment_length) for _ in range(num_segments)]
    
    feature_matrix = extract_features(data_segments)
    
    print("Feature matrix shape:", feature_matrix.shape)


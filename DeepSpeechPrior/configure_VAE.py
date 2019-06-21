GPU = 0 # the ID of GPU
LEARNING_RATE = 1e-3 # learning rate of Adam
N_EPOCH = 200        # the number of epoch
BATCH_SIZE = 1000    # batch size
N_FFT = 1024         # window length of STFT
HOP_LENGTH = 256     # shift width of STFT
N_LATENT = 16        # dimension of latent variable

WSJ0_PATH = "./data/audio/16kHz/isolated/tr05_org/" # directory in which clean wav files exist
DATASET_SAVE_PATH = "/home/sekiguchi/data/" # directory to save the dataset which is made from WSJ0_PATH
MODEL_SAVE_PATH = "./" # directory to save a trained VAE

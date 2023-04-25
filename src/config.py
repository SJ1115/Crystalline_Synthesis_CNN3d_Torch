class config:
    #### Model ####
    ## input size
    input_size = 128
    input_channel = 3
    ## CNN
    kernel_size = 3
    stride = 1
    padding = 1
    ## model's
    dropout = .3
    
    #### RUN ####
    is_unsupervised = False
    L2 = 3
    epoch = 20
    batch_size = 16
    device = "cuda:0"
    use_tqdm = True
    use_board = False
    train = "train.pkl"
    dev   = "dev.pkl"
    test  = "test.pkl"

class config_CNN(config):
    ## CNN
    channels = [32, 32, 32]
    pool = [4, 4, 4]
    ## Model
    dropout = 0
    
class config_CAE(config):
    channels = [32, 32, 64]
    pool = [4, 4, 2]

    is_unsupervised = True
    batch_size = 8

"""For model_re.py
"""
class config_CNN_re(config):
    ## CNN
    channels = [32, 32, 32]
    pool = [4, 4, 4]
    ## Model
    dropout_conv = 0
    dropout_FC = .5
    
    batch_size=8

class config_CAE_re(config):
    channels = [32, 32, 64]
    pool = [4, 4, 2]

    dropout_FC = .5
    is_unsupervised = True
    batch_size = 8

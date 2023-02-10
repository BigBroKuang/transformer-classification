import os 
import torch

class Config():
    #initialize parameters
    def __init__(self):
        #path of the directory
        self.proj_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.dataset_dir = os.path.join(self.proj_dir,'data')
        #proj_path/data/ag_news_csv/
        self.train_data_paths = os.path.join(self.dataset_dir, 'ag_news_csv', 'train.csv')  
        self.test_data_paths = os.path.join(self.dataset_dir, 'ag_news_csv', 'test.csv')
        
        self.min_freq = 1
        self.max_sen_len =None
        self.warmup = 4000
        #model parameters
        self.batch_size = 128
        self.d_model = 512
        self.num_head = 8 #512/8 = 64
        self.num_encoder_layers = 6
        self.num_decoder_layers = 6
        self.dim_feedforward = 512 #dim of FF layer
        # self.num_classification = 256 #layer before the output
        self.dim_classification = 256
        self.num_class = 4
        self.dropout = 0.1
        self.concat_type = 'avg'
        self.beta1 = 0.9
        self.beta2 = 0.98
        self.epsilon = 1e-9
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.epoch = 10
        self.model_save_dir = os.path.join(self.proj_dir, 'cache')
        self.model_save_per_epoch = 2
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
            
#test 
if __name__ == '__main__':
    config = Config()
    print(config.proj_dir)
    print(config.train_data_paths)        
        
        
        
        
        
        
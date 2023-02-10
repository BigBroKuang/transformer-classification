from collections import Counter
from torchtext.vocab import vocab
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
import torch
import re
from tqdm import tqdm


def clean_str(string):
    #replace signs (except letters, numbers, -, ?, !, ., ,) with space
    string = re.sub("[^A-Za-z0-9\-\?\!\.\,]", " ", string).lower()
    return string
    
def build_vocab(filepath, tokenizer, min_freq):
    #special signs
    specials = ['<unk>', '<pad>']
    counter = Counter()
    with open(filepath, encoding = 'utf8') as f:
        for string in tqdm(f):
            string = string.strip().split('","')[-1][:-1]
            #print(string)
            counter.update(tokenizer(clean_str(string)))
    #build vocab
    return vocab(counter, min_freq=min_freq, specials=specials)

def pad_sequence(sequences, max_len=None, pad_value =0):
    #pad the sentences
    #use the max length of all sentences or the batch
    max_len_all = max([s.size(0) for s in sequences])
    if max_len is not None:
        max_len = max(max_len, max_len_all)
    else:
        max_len = max_len_all
    
    #create a zero tensor
    pad_tensor = torch.zeros((len(sequences), max_len))
    #fill the zero tensor with sequence 
    for idx, tensor in enumerate(sequences):
        length = tensor.size(0)
        pad_tensor[idx,:length] = tensor
    return pad_tensor
    
class LoadSentenceClassificationDataset():
    def __init__(self, config):
        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = build_vocab(config.train_data_paths,self.tokenizer, config.min_freq)
        #print(self.vocab['<unk>'],self.vocab['is'])
        self.PAD_IDX = self.vocab['<pad>']
        self.vocab.set_default_index(self.vocab['<unk>'])#unseen words are default to <unk>
        self.config = config
        
    def data_process(self,filepath):
        #create training and testing datasets
        #data -> [([seq1], class),([seq2], class) ,([seq3], class)]
        #words in seqs are replaced with indexes 
        raw_input = open(filepath, encoding='utf8').readlines()
        data = []
        max_len = 0
        for raw_in in tqdm(raw_input, ncols=80):
            line = raw_in.rstrip('\n').split('","')
            s,l = clean_str(line[-1][:-1]),line[0][-1]
            #print(s)
            seq_ = torch.tensor([self.vocab[tok] for tok in self.tokenizer(s)], dtype=torch.long)
            l = torch.tensor(int(l)-1, dtype=torch.long)
            max_len = max(max_len, seq_.size(0))
            data.append((seq_, l))
            #[(sentence, label), (sentence, label), ...]
        return data, max_len
            
            

    def load_train_test_data(self,config):

        train_data, max_sen = self.data_process(config.train_data_paths)
        if not config.max_sen_len:
            self.max_sen_len = max_sen 
        test_data, _ = self.data_process(config.test_data_paths)
        #self.generate_batch(train_data)
        #generate training batches with dataloader
        #collate_fn is rewritten since the lengths of seqs are not the same
        
        train_iter = DataLoader(train_data, batch_size=config.batch_size, collate_fn=self.generate_batch)
        test_iter = DataLoader(test_data, batch_size=config.batch_size, collate_fn=self.generate_batch)
        
        # for i,d in enumerate(train_iter):
        #     print(i, d[0].size())
        return train_iter, test_iter
    
    def generate_batch(self, data_in):
        
        batch_sen, batch_label =[],[]
        for sen, label in data_in:
            batch_sen.append(sen)
            batch_label.append(label)
        #print('# training sequences:', len(batch_sen))
        #pad the seqs 
        batch_sen = pad_sequence(batch_sen, self.config.max_sen_len)
        batch_label = torch.tensor(batch_label, dtype =torch.long)
        return batch_sen, batch_label
        

        
        
        
        
        
        
        
        
        
        
        
        
        
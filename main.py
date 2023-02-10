import torch.nn as nn
import torch
from config import Config
from utils import LoadSentenceClassificationDataset
from ClassificationModel import ClassificationModel
import os
import time
from copy import deepcopy

class traing_setting(nn.Module):
    def __init__(self, config):
        super(traing_setting, self).__init__()
        self.d_model  = torch.tensor(config.d_model)
        self.warmup = config.warmup
        self.step = 1.0
    def __call__(self):
        arg1 = self.step**(-0.5)
        arg2 = self.step*(self.warmup**(-1.5))
        self.step +=1.
        return (self.d_model**-0.5)*min(arg1, arg2)
        
def train_model(config):
    
    data_loader = LoadSentenceClassificationDataset(config) #load data
    train_data, test_data = data_loader.load_train_test_data(config) #build training/testing dataloader
    
    classification_model = ClassificationModel(vocab_size =len(data_loader.vocab), config = config)#build model
    
    model_save_path = os.path.join(config.model_save_dir, 'model.pt') #model saving path
    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path)
        classification_model.load_state_dict(loaded_paras)    
        
    classification_model = classification_model.to(config.device) #traing the model in cuda/cpu
    loss_fn = torch.nn.CrossEntropyLoss() #loss function
    learning_rate = traing_setting(config) #set learning rate
    optimizer = torch.optim.Adam(classification_model.parameters(), lr=0., betas = (config.beta1, config.beta2), eps=config.epsilon)
    
    classification_model.train()
    
    for epoch in range(config.epoch):
        losses = 0
        start_time = time.time()
        for idx, (sample, label) in enumerate(train_data):
            sample = sample.to(config.device)
            label = label.to(config.device)
            pad_mask = (sample==data_loader.PAD_IDX) #(batch, seq_len)
            logits = classification_model(sample, key_pad_mask =pad_mask)
            
            optimizer.zero_grad()
            loss = loss_fn(logits, label)
            loss.backward()
            lr = learning_rate()
            for p in optimizer.param_groups:
                p['lr'] = lr
            optimizer.step()
            losses += loss.item()
            
            acc = (logits.argmax(1) == label).float().mean()
            if idx % 10 == 0:
                print(f"Epoch: {epoch}, Batch[{idx}/{len(train_data)}], "
                      f"Loss :{loss.item():.3f}, Train acc: {acc:.3f}")
        end_time = time.time()
        train_loss = losses / len(train_data)
        print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")
        if (epoch + 1) % config.model_save_per_epoch == 0:
            acc = evaluate(test_data, classification_model, config)
            print(f'Epoch:{epoch}, accuracy on test {acc:.3f}')
            
def evaluate(test_data, model, config):
    model.evaulate()
    acc = 0
    n_sample = 0
    with torch.no_grad():

        for sample, label in test_data:
            sample = sample.to(config.device)
            label = label.to(config.device)
            logits = model(sample)
            acc +=  (logits.argmax(1)==label).float().sum().item() 
            n_sample += len(label)
        model.train()
    return acc/n_sample

if __name__ == '__main__':
    config = Config()
    train_model(config)
    
    
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle

from sklearn.metrics import f1_score
from datetime import timedelta
from time import time
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


class Molecule_Dataset(Dataset):
    def __init__(self, filename):
        with open(filename, "rb") as f:
            self.data = pickle.load(f,)

        self.len = len(self.data)
                
    def __getitem__(self, idx):
        image = torch.tensor(self.data[idx]['image']).transpose(0,3).to(torch.float)
        label = torch.tensor(self.data[idx]['label']).to(torch.float)
        return image, label
    
    def __len__(self):
        return self.len

class Trainer:
    def __init__(self, model, criterion, optimizer, config):
        self.model = model.to(config.device)
        self.criterion = criterion
        self.optimizer = optimizer
        
        self.L2 = config.L2
        self.epoch = config.epoch
        
        self.trainLoader = DataLoader(Molecule_Dataset(config.train), batch_size = config.batch_size, shuffle=True)
        self.devLoader   = DataLoader(Molecule_Dataset(config.dev  ), batch_size = config.batch_size)
        self.testLoader  = DataLoader(Molecule_Dataset(config.test ), batch_size = config.batch_size)

        self.loss_list = []
        
        self.device = config.device
        self.epoch = config.epoch
        
        self.is_unsupervised = config.is_unsupervised

        self.verbose = config.use_tqdm
        self.verboard = config.use_board
        if self.verboard:
            self.writer = SummaryWriter(model._get_name() + "/")
        
    def train(self, show_batches=1):
        if self.verbose:
            start = time()
            timeiter = tqdm(total = len(self.trainLoader) * self.epoch, position=0, leave=True)

        # Set Train Mode
        self.model.train()
        torch.enable_grad()

        if self.verboard:
            runs = 0
        
        for epoch in range(self.epoch):  # loop over the dataset multiple times

            running_loss = 0.0
            cut = 0
            for data in self.trainLoader:
                # get the inputs; data is a tuple of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                if self.is_unsupervised:
                    labels = inputs
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model.forward(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.L2)
                self.optimizer.step()

                running_loss += loss.item()
                cut += 1
                (lambda x: timeiter.update(1) if x else 0)(self.verbose)
                if cut % show_batches == 0:
                    current_loss = running_loss / show_batches
                    self.loss_list.append(current_loss)
                    running_loss = 0.0
                    cut = 0

                    if self.verbose:
                    # print statistics        
                        timeiter.set_description(f"loss : {current_loss: .3f}")
                    if self.verboard:
                        runs += 1
                        self.writer.add_scalar("loss", current_loss, runs)
        
        if self.verbose:
            timeiter.close() 
            print(f"Finished Training : {str(timedelta(seconds=int(time() - start), ))} spent.")
        if self.verboard:
            self.writer.close()
        return
    
    def test(self, test=True, f1_method = 'binary'):
        # ACC
        correct = 0
        total = 0
        # F1
        ans  = []
        pred = []
        assert f1_method in ('micro', 'macro', 'binary')
        
        if test:
            Loader = self.testLoader
        else:
            Loader = self.devLoader
        
        # since we're not training, we don't need to calculate the gradients for our outputs
        self.model.eval()
        with torch.no_grad():
            for data in Loader:
                inputs, labels = data                
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # calculate outputs by running images through the network
                outputs = self.model.forward(inputs)
                # the class with the highest energy is what we choose as prediction
                predicted = outputs.data.round()
                
                # ACC
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # F1
                ans  += list(labels.to('cpu'))
                pred += list(predicted.to('cpu'))

        f1 = f1_score(ans, pred, average=f1_method)
        if self.verbose:
            print(f'Test on the {len(self.testLoader.dataset)} set')
            print(f'Accuracy : {100 * correct / total:.2f} %') 
            print(f'F1 Score : {100 * f1 : .2f}')
        return correct/total, f1

    def plot(self):
        if len(self.loss_list)==0:
            raise ValueError("Train Model First")
        
        fig = plt.plot(self.loss_list)

        plt.show(fig)
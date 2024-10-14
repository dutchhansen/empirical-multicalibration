import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
from datetime import datetime
import os


class _general_MLP(nn.Module):
    def __init__(self, layer_widths):
        '''
        layer_widths must be a list of integers, where the first element is the 
        input dimension and the last element is the output dimension
        '''
        super(_general_MLP, self).__init__()
        self.layers = []

        # Write a loop which adds layers of width layer_widths[i] to the network
        # However, if the width is 'BN', add a batch norm layer instead.
        # Furthermore, after every linear layer, add a ReLU layer, except for the last layer.
        # If the last layer is 'BN', raise an error.
        for i in range(1, len(layer_widths)):
            if isinstance(layer_widths[i], int):
                if layer_widths[i-1] == 'BN':
                    self.layers.append(nn.Linear(layer_widths[i-2], layer_widths[i]))
                else:
                    self.layers.append(nn.Linear(layer_widths[i-1], layer_widths[i]))
                if i < len(layer_widths) - 1:
                    self.layers.append(nn.ReLU())
            elif layer_widths[i] == 'BN':
                if i == len(layer_widths) - 1:
                    raise ValueError('Invalid layer width list, batch norm layer cannot be the last layer')
                self.layers.append(nn.BatchNorm1d(layer_widths[i-1]))
            else:
                raise ValueError('Unknown layer type or invalid layer width list')
        
        # Make parameters findable by pytorch
        self.layers = nn.ModuleList(self.layers)

        
    def forward(self, x):
        for idx in range(len(self.layers)-1):
            x = self.layers[idx](x)
        x = self.layers[-1](x)

        return x


class MLP:
    def __init__(self, SAVE_DIR, config, from_saved=False):
        """
        Simple neural network class
        """
        self.config = config
        self.SAVE_DIR = SAVE_DIR
        self.from_saved = from_saved

        # init model
        self.load_config(self.config)
        self.net = self.load_net(self.arch, self.from_saved)
        

    def train(self, X_train, y_train, groups_train, X_val, y_val, groups_val):
        """
        Train the neural network
        """
        # if SAVE_DIR is not a directory yet, create it
        if not os.path.exists(self.SAVE_DIR):
            os.makedirs(self.SAVE_DIR)

        # define the loss function and the optimiser
        self.criterion = nn.CrossEntropyLoss()
        if self.optim_name == 'sgd':
            self.optimizer = optim.SGD(self.net.parameters(), 
                                       lr=self.lr_schedule[0], 
                                       weight_decay=self.weight_decay,
                                       momentum=self.momentum)
        elif self.optim_name == 'adam':
            self.optimizer = optim.Adam(self.net.parameters(), 
                                        lr=self.lr_schedule[0], 
                                        weight_decay=self.weight_decay)
        else:
            raise ValueError('Unknown optimizer')
        
        # move data to device
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(self.device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(self.device)

        # Make train dataloader
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        train_loss = []
        train_accs = []
        best_val_acc = -1

        # determine epochs
        if self.from_saved:
            if self.saved_epoch >= self.epochs - 1:
                raise ValueError('Model already trained for', self.saved_epoch, 'epochs.')
            
            epoch_range = range(self.saved_epoch + 1, self.epochs)

            # find current learning rate
            for lr_epoch in self.lr_schedule:
                if lr_epoch <= self.saved_epoch:
                    current_lr = self.lr_schedule[lr_epoch]
                else: break

        else: epoch_range = range(self.epochs)

        # train loop
        for epoch in epoch_range:
            running_loss = 0.0

            if epoch in self.lr_schedule:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr_schedule[epoch]
                    current_LR = param_group['lr']

            for i, data in enumerate(train_dataloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                # data already on device
                
                # zero the parameter gradients
                self.optimizer.zero_grad()
                
                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 100 == 0:    # print every 1000 mini-batches
                    train_loss.append(running_loss / 1000)
                    print('[%d, %5d] loss: %.3f LR: %.5f' %
                        (epoch + 1, i + 1, running_loss / 1000, current_LR))

            self.net.eval()
            with torch.no_grad():
                train_loss.append(running_loss)
                y_pred = np.argmax(self.net(X_train_tensor).detach().cpu().numpy(), axis=1)
                train_acc = accuracy_score(y_train_tensor.cpu().numpy(), y_pred)
                print('train acc', train_acc)
                train_accs.append(train_acc)

                # validate every val_eval_epoch epochs
                if epoch % self.val_eval_epoch == 0:
                    outputs = self.net(X_val_tensor)
                    y_pred = np.argmax(outputs.detach().cpu().numpy(), axis=1)

                    # Check fraction of 1s in validation data and predictions
                    print('y_val mean pred', np.mean(y_pred))
                    print('y true mean', np.mean(y_val_tensor.cpu().numpy()))

                    val_acc = accuracy_score(y_val_tensor.cpu().numpy(), y_pred)
                    print('val acc', val_acc)

                    if best_val_acc < val_acc and epoch >= self.val_save_epoch:
                        best_val_acc = val_acc
                        torch.save(self.net.state_dict(), self.SAVE_DIR + 'model.pt')

        # save config to file
        with open(self.SAVE_DIR + 'model_config.txt', 'w') as f:
            f.write(str(self.config))

        # Load the best model
        print('Best validation accuracy:', best_val_acc)
        self.net.load_state_dict(torch.load(self.SAVE_DIR + 'model.pt'))
    
    def predict_proba(self, X, with_logits=False):
        X_test_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        # convert predictions to probabilities
        logit = self.net(X_test_tensor).detach().cpu()
        p = torch.nn.functional.softmax(logit, dim=1).numpy()

        if with_logits: return p, logit.numpy()
        else: return p

    def predict(self, X):
        X_test_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        nn_preds = np.argmax(self.net(X_test_tensor).detach().cpu().numpy(), axis=1)

        return nn_preds
    
    def load_net(self, arch, from_saved):
        net = _general_MLP(arch)
        if from_saved:
            net.load_state_dict(torch.load(self.SAVE_DIR + 'model.pt'))
        return net.to(self.device)
        

    def load_config(self, config):
        """
        Load model configuration.
        """
        self.arch = config['arch']
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.lr_schedule = config['lr_schedule']
        self.optim_name = config['optim']
        self.weight_decay = config['weight_decay']
        self.momentum = config['momentum']

        # require momentum for SGD
        assert self.optim_name != 'sgd' or self.momentum is not None, 'Momentum must be specified for SGD optimizer'

        # sanity check
        if 0 not in self.lr_schedule:
            raise ValueError('Learning rate schedule must start at / contain epoch 0')
        if config['val_save_epoch'] > config['epochs'] - 1:
            raise ValueError(('Note: val_save_epoch must be <= (# epochs - 1); ' +
                              'model only saved when (# epochs elapsed) > val_save_epoch.'))
        
        # Determines after how many epochs we start saving the model based on validation accuracy
        self.val_save_epoch = config['val_save_epoch']
        # Determines how often to evaluate the validation accuracy
        self.val_eval_epoch = config['val_eval_epoch']
        
        self.device = "cpu"
        # check cuda
        if torch.cuda.is_available():
            self.device = "cuda"
            print("Setting device = cuda.")
        # check for MPS (Mac M1)
        elif torch.backends.mps.is_available():
            self.device = "mps"
            print("Setting device = MPS.")
        elif not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not "
                    "built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine.")
        # cpu default
        else:
            print("No device found; setting device = cpu.")

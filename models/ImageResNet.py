import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
from dataloaders.utils.image_utils import ImageDatasetWrapper
from configs.datasets import dataset_info
import torchvision
import os

SUPPORTED_NNS = {
    'resnet18': torchvision.models.resnet18,
    'resnet34': torchvision.models.resnet34,
    'resnet50': torchvision.models.resnet50,
    'resnet101': torchvision.models.resnet101,
    'wideresnet50': torchvision.models.wide_resnet50_2,
    'densenet121': torchvision.models.densenet121
}

class ImageResNet:
    def __init__(self, SAVE_DIR, config, from_saved=False):
        self.config = config
        self.SAVE_DIR = SAVE_DIR
        self.from_saved = from_saved
        self.load_config(self.config)

    def train(self, X_train, y_train, groups_train, X_val, y_val, groups_val):
        if self.config is None:
            raise ValueError('Config not provided.')
        
        # if SAVE_DIR is not a directory yet, create it
        if not os.path.exists(self.SAVE_DIR):
            os.makedirs(self.SAVE_DIR)

        # build dataset and dataloader objects
        train_dataset = ImageDatasetWrapper(X_train, y_train, self.dataset_config)
        val_dataset = ImageDatasetWrapper(X_val, y_val, self.dataset_config)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                       batch_size=self.batch_size, 
                                                       shuffle=True,
                                                       num_workers=self.NUM_WORKERS,
                                                       pin_memory=self.PIN_MEMORY,
                                                       drop_last=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=self.batch_size,
                                                     shuffle=False,
                                                     num_workers=self.NUM_WORKERS,
                                                     pin_memory=self.PIN_MEMORY,
                                                     drop_last=False)

        # define loss function and optimiser
        criterion = nn.CrossEntropyLoss().to(self.device)
        if self.config['optim'] == 'sgd':
            optimizer = optim.SGD(self.net.parameters(), 
                                  lr=self.lr_schedule[0], 
                                  weight_decay=self.weight_decay)
        elif self.config['optim'] == 'adam':
            optimizer = optim.Adam(self.net.parameters(), 
                                   lr=self.lr_schedule[0], 
                                   weight_decay=self.weight_decay)
        else:
            raise ValueError('Optimizer not supported.')
        
        train_loss = []
        train_accs = []
        best_val_acc = -1

        # train loop
        for epoch in range(self.epochs):
            self.net.train()
            running_loss = 0.0
            running_acc = 0.0
            num_batches = 0

            # update learning rate
            if epoch in self.lr_schedule:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.lr_schedule[epoch]
                    current_lr = param_group['lr']

            for i, data in enumerate(train_dataloader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # backprop
                optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                outcopy = outputs.detach().cpu().numpy()
                labelscopy = labels.detach().cpu().numpy()
                curr_acc = accuracy_score(np.argmax(outcopy, axis=1), labelscopy)

                num_batches += 1
                running_acc += curr_acc

                # print statistics
                running_loss += loss.item()
                if i % 100 == 0:    # print every 1000 mini-batches
                    train_loss.append(running_loss / 100)
                    print('[%d, %5d] loss: %.3f LR: %.5f running acc: %.5f' %
                        (epoch + 1, i + 1, running_loss / 100, current_lr, running_acc / num_batches))
                    
            print('Epoch:', epoch + 1, 'Loss:', running_loss, 'Running acc:', running_acc / num_batches)

            # validate every val_eval_epoch epochs
            if epoch % self.val_eval_epoch == 0:
                self.net.eval()
                with torch.no_grad():

                    ## validation split
                    # get predictions
                    (y_conf, y_logits), y_val = self.evaluate_loader(val_dataloader)
                    y_pred = torch.argmax(y_conf, axis=1).cpu().numpy()
                    y_val = y_val.cpu().numpy()

                    # metrics
                    print('val: y-pred mean', np.mean(y_pred))
                    print('val: y-true mean', np.mean(y_val))
                    val_acc = accuracy_score(y_val, y_pred)
                    print('val acc', val_acc)

                    if best_val_acc < val_acc and epoch >= self.val_save_epoch:
                        best_val_acc = val_acc
                        torch.save(self.net.state_dict(), self.SAVE_DIR + 'model.pt')

        # save config to file
        with open(self.SAVE_DIR + 'config.txt', 'w') as f:
            f.write(str(self.config))

        # load best model
        print('Best validation accuracy:', best_val_acc)
        self.net.load_state_dict(torch.load(self.SAVE_DIR + 'model.pt'))

    def evaluate_loader(self, dataloader):
        '''
        Evaluate model on dataloader, return detatched confidence/labels.
        '''
        all_confs, all_logits, all_lbls = [], [], []

        for i, batch in enumerate(dataloader):
            inputs, labels = batch
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.net(inputs).detach()
            p = F.softmax(outputs, dim=1)

            all_logits.append(outputs)
            all_confs.append(p)
            all_lbls.append(labels)

        return (torch.cat(all_confs), torch.cat(all_logits)), torch.cat(all_lbls)
    
    def predict_proba(self, X_idxs, with_logits=False):
        '''
        Prediction function for evaluation.
        Used only by Experiment.py class.
        '''
        ds = ImageDatasetWrapper(X_idxs, y=None, config=self.dataset_config)
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, 
                                         shuffle=False,
                                         num_workers=self.NUM_WORKERS,
                                         pin_memory=self.PIN_MEMORY,
                                         drop_last=False)
        
        (confs, logits), labels = self.evaluate_loader(dl)
        if with_logits: return confs.cpu().numpy(), logits.cpu().numpy()
        else: return confs.cpu().numpy()

    def predict(self, X_text):
        '''
        Prediction function for evaluation.
        Used only by Experiment.py class.
        '''
        confs = self.predict_proba(X_text)
        preds = np.argmax(confs, axis=1)

        return preds

    def load_net(self, arch, from_saved):
        init_net = SUPPORTED_NNS[arch]
        net = init_net(weights=self.pretrained)

        # change last layer
        if arch == 'densenet121':
            net.classifier = nn.Linear(net.classifier.in_features, 2)
        else:
            net.fc = nn.Linear(net.fc.in_features, 2)
        
        # check if saved
        if from_saved:
            net.load_state_dict(torch.load(self.SAVE_DIR + 'model.pt'))
            print('Loaded model from:', self.SAVE_DIR + 'model.pt')
        
        return net.to(self.device)

    def load_config(self, config):
        """
        Load model configuration.
        """
        self.arch = config['arch']
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.lr_schedule = config['lr_schedule']
        self.momentum = config['momentum']
        self.dataset_config = dataset_info(config['dataset'])
        self.pretrained = config['pretrained']
        self.NUM_WORKERS = 2
        self.PIN_MEMORY = True

        # sanity check on learning rate schedule
        if 0 not in self.lr_schedule:
            raise ValueError('Learning rate schedule must start at / contain epoch 0')
        if config['val_save_epoch'] > config['epochs'] - 1:
            raise ValueError(('Note: val_save_epoch must be <= (# epochs - 1); ' +
                              'model only saved when (# epochs elapsed) > val_save_epoch.'))
        
        # Determines after how many epochs we start saving the model based on validation accuracy
        self.val_save_epoch = config['val_save_epoch']
        # Determines how often to evaluate the validation accuracy
        self.val_eval_epoch = config['val_eval_epoch']
        
        # init device
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

        # init network
        self.net = self.load_net(self.arch, self.from_saved)

        # init loss and optimizer
        self.weight_decay = config['weight_decay']
        self.criterion = nn.CrossEntropyLoss()
        if config['optim'] == 'sgd':
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr_schedule[0], 
                                       momentum=self.momentum, weight_decay=self.weight_decay)
        elif config['optim'] == 'adam':
            if 'momentum' in config and config['momentum'] != 0:
                raise ValueError('Momentum not supported for Adam optimizer.')
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr_schedule[0], 
                                        weight_decay=self.weight_decay)
        else:
            raise ValueError('Unknown optimizer')

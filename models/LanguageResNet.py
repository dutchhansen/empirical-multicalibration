import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
import transformers
import numpy as np
from sklearn.metrics import accuracy_score
from dataloaders.utils.language_utils import LanguageDatasetWrapper
from torchmetrics import Accuracy, CalibrationError
import os
import shutil

'''
ResNet for text classification.

@article{RNTI/papers/1002807,
    author    = {Corentin Duchêne and Henri Jamet and Pierre Guillaume and Réda Dehak},
    title     = {Benchmark pour la classification de commentaires toxiques sur le jeu de données Civil Comments},
    journal = {Revue des Nouvelles Technologies de l'Information},
    volume = {Extraction et Gestion des Connaissances, RNTI-E-39},
    year      = {2023},
    pages     = {19-30}
}

Original implementation from: 
https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py

'''

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    

class _ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels=3, num_classes=10):
        super(_ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, kernel_size=(out.size()[2], out.size()[3]))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class _LanguageResNet(nn.Module):
        def __init__(self, config, dataset_obj=None):
            super(_LanguageResNet, self).__init__()
            assert dataset_obj is not None, 'Please provide dataset object.'
            self.vocab  = self.load_vocab(dataset_obj, 
                                          tokenizer=config['tokenizer'], 
                                          max_token_len=config['max_token_len'], 
                                          min_freq=config['min_freq'])
            self.load_resnet(config)


        def load_vocab(self, dataset_obj, tokenizer, max_token_len, min_freq):
            '''
            Load vocabulary.
            '''
            vocab = dataset_obj.vocab(tokenizer, max_token_len, min_freq)
            return vocab
            

        def load_resnet(self, config):
            '''
            Configure network.
            '''
            ( # unpack config
                in_channels,
                pretrained_embedding,
                resnet_type,
                embedding_dim,
                freeze_embedding,
                stack_embedding,
                tokenizer,
                max_token_len
            ) = (
                config['in_channels'],
                config['pretrained_embedding'],
                config['resnet_type'],
                config['embedding_dim'],
                config['freeze_embedding'],
                config['stack_embedding'],
                config['tokenizer'],
                config['max_token_len']
            )

            # If true stack embedding to 3 channels
            self.stack_embedding = stack_embedding

            # constants
            vocab = self.vocab
            vocab_size = len(vocab)
            pad_index = vocab['<pad>']

            # Embedding type
            if pretrained_embedding == "glove":
                self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)

                # Download pretrained glove embedding
                vectors = torchtext.vocab.GloVe()

                # Get the pretrained embedding vectors according to the vocabulary
                pretrained_embedding = vectors.get_vecs_by_tokens(vocab.get_itos())
                self.embedding.weight.data = pretrained_embedding

            elif pretrained_embedding == "fasttext":
                self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)

                # Download pretrained FastText embedding
                vectors = torchtext.vocab.FastText()

                # Get the pretrained embedding vectors according to the vocabulary
                pretrained_embedding = vectors.get_vecs_by_tokens(vocab.get_itos())
                self.embedding.weight.data = pretrained_embedding

            elif pretrained_embedding == "roberta":
                # does not work admit sequence length > 512
                # Download the pretrained Transformer model
                tr_model = transformers.AutoModel.from_pretrained('roberta-base')

                # set the embedding parameters
                self.embedding = tr_model.embeddings
            
            else:
                self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)

            resnet_types_dic = {"20": 0, "32": 1, "44": 2, "56": 3, "110": 4, "1202": 5}
            resnet_params = [[3, 3, 3], [5, 5, 5], \
                            [7, 7, 7], [9, 9, 9], \
                            [18, 18, 18], [200, 200, 200]]

            if resnet_type not in resnet_types_dic:
                print("Error: 'resnet_type' should be in: ['20', '32', '44', '56', '110', '1202']")

            resnet_param_idx = resnet_types_dic[resnet_type]
            self.net = _ResNet(BasicBlock,
                                    resnet_params[resnet_param_idx],
                                    in_channels=in_channels,
                                    num_classes=2)

            # freeze all the embedding parameters if necessary
            for param in self.embedding.parameters():
                param.requires_grad = not freeze_embedding

        def forward(self, ids):
            x = self.embedding(ids)
            if self.stack_embedding:
                # Transform the image in 3 channels
                split = x.shape[-1] // 3
                x = torch.stack([x[..., 0:split], x[..., split:2*split], x[..., 2*split:]], dim=1)
            else:
                x = torch.unsqueeze(x, 1)
            x = self.net(x)

            return x


class LanguageResNet:
    def __init__(self, SAVE_DIR, config, from_saved=False, saved_epoch=0, dataset_obj=None, save_scheme='best-val-acc'):
        assert dataset_obj is not None, 'Please provide dataset object.'
        self.SAVE_DIR = SAVE_DIR
        self.saved_epoch = saved_epoch
        self.from_saved = from_saved
        self.save_scheme = save_scheme
        self.load_config(config)

        # load model and training objects
        self.net = self.load_net(config, dataset_obj, from_saved, saved_epoch)
        (self.vocab, self.tokenizer, self.max_token_len) = (self.net.vocab, config['tokenizer'], config['max_token_len'])

        # data transform
        self.transform_config = {
            'model': 'ResNet',
            'tokenizer': self.tokenizer,
            'vocab': self.vocab,
            'max_token_len': self.max_token_len
        }

        print("ResNet model loaded. Embedding:", config['pretrained_embedding'])

    def load_net(self, config, dataset_obj, from_saved, saved_epoch=0):
        net = _LanguageResNet(config, dataset_obj).to(self.device)
        if from_saved:
            net.load_state_dict(torch.load(self.SAVE_DIR + f'model_{saved_epoch}.pt'))
            print(f'Loaded model from epoch {self.SAVE_DIR}model_{saved_epoch}.pt')
        return net.to(self.device)

    def train(self, X_train, y_train, groups_train, X_val, y_val, groups_val):
        
        # if SAVE_DIR is not a directory yet, create it
        if not os.path.exists(self.SAVE_DIR):
            os.makedirs(self.SAVE_DIR)

        # build dataset and dataloader objects
        train_dataset = LanguageDatasetWrapper(X_train, y_train, config=self.transform_config)
        val_dataset = LanguageDatasetWrapper(X_val, y_val, config=self.transform_config)
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
        if self.optim == 'sgd':
            optimizer = optim.SGD(self.net.parameters(), 
                                  lr=self.lr_schedule[0], 
                                  weight_decay=self.weight_decay)
        elif self.optim == 'adam':
            optimizer = optim.Adam(self.net.parameters(), 
                                   lr=self.lr_schedule[0], 
                                   weight_decay=self.weight_decay)
        else:
            raise ValueError('Optimizer not supported.')
        
        train_loss = []
        best_val_acc = -1
        accuracy = Accuracy(task='multiclass', num_classes=2).to(self.device)
        ece = CalibrationError(task='multiclass', num_classes=2, n_bins=10).to(self.device)

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
        self.recent_save = None
        for epoch in epoch_range:
            self.net.train()
            running_loss = 0.0
            running_acc = 0.0
            running_ece = 0.0
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

                out_d = F.softmax(outputs.detach(), dim=1)
                labels_d = labels.detach()
                curr_acc = accuracy(out_d, labels_d)
                curr_ece = ece(out_d, labels_d)

                num_batches += 1
                running_acc += curr_acc
                running_ece += curr_ece

                # print statistics
                running_loss += loss.item()
                if i % 100 == 0:    # print every 1000 mini-batches
                    train_loss.append(running_loss / 100)
                    print('[%d, %5d] loss: %.3f LR: %.5f running acc: %.5f running ECE %.5f' %
                        (epoch + 1, i + 1, running_loss / 100, 
                         current_lr, running_acc / num_batches, 
                         running_ece / num_batches))
                    
            print('Epoch:', epoch + 1, 'Loss:', running_loss, 
                  'Running acc:', running_acc / num_batches,
                  'Running ECE:', running_ece / num_batches)

            # validate every val_eval_epoch epochs
            if epoch % self.val_eval_epoch == 0:
                self.net.eval()
                with torch.no_grad():

                    ## validation split
                    # get predictions
                    (y_conf, y_logits), y_val = self.evaluate_loader(val_dataloader)
                    y_pred = torch.argmax(y_conf, axis=1)

                    # metrics
                    print('val: y-pred mean', round(torch.mean(y_pred.type(torch.float32)).item(), 3))
                    print('val: y-true mean', round(torch.mean(y_val.type(torch.float32)).item(), 3))
                    val_acc = accuracy(y_val, y_pred).item()
                    print('val: acc', round(val_acc, 3))

                    # save every epoch, according to save_scheme
                    self.save_model(epoch, val_acc, best_val_acc)
                    best_val_acc = max(val_acc, best_val_acc)

        # save config to file
        with open(self.SAVE_DIR + 'config.txt', 'w') as f:
            f.write(str(self.config))

        # load best model
        print('Best validation accuracy:', best_val_acc)
        self.net.load_state_dict(torch.load(self.SAVE_DIR + f'model_{self.recent_save}.pt'))

    def save_model(self, epoch, val_acc, old_best_val_acc):
        '''save model according to save_scheme'''
        if epoch < self.val_save_epoch: return
        elif self.recent_save == None:
            self.recent_save = epoch
            torch.save(self.net.state_dict(), self.SAVE_DIR + f'model_{epoch}.pt')
            return
        
        # all-epochs: save every epoch
        if self.save_scheme == 'all-epochs':
            self.recent_save = epoch    
            torch.save(self.net.state_dict(), self.SAVE_DIR + f'model_{epoch}.pt')

        # best-val-acc: save only if val_acc is better than old_best_val_acc
        elif self.save_scheme == 'best-val-acc':
            self.recent_save = epoch
            if val_acc > old_best_val_acc:
                torch.save(self.net.state_dict(), self.SAVE_DIR + f'model_{epoch}.pt')
            else:
                shutil.copy(self.SAVE_DIR + f'model_{epoch - self.val_eval_epoch}.pt', self.SAVE_DIR + f'model_{epoch}.pt')

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

    def load_config(self, config):
        '''
        Load model configuration.
        '''
        self.config = config

        (
            self.optim,
            self.epochs,
            self.batch_size,
            self.lr_schedule,
            self.weight_decay,
            self.val_save_epoch,
            self.val_eval_epoch
        ) = (
            config['optim'],
            config['epochs'],
            config['batch_size'],
            config['lr_schedule'],
            config['weight_decay'],
            config['val_save_epoch'],
            config['val_eval_epoch']
        )

        # training
        self.NUM_WORKERS = 2
        self.PIN_MEMORY = True
        
        # determine device
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
        
    def predict_proba(self, X_text, with_logits=False):
        '''
        Prediction function for evaluation.
        Used only by Experiment.py class.
        '''
        ds = LanguageDatasetWrapper(X_text, y=None, config=self.transform_config)
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
    
    
    
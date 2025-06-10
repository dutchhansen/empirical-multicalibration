import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
from datetime import datetime
import os


class _general_MLP(nn.Module):
    def __init__(self, layer_widths, num_groups=None, embedding_dim=None):
        '''
        layer_widths must be a list of integers, where the first element is the 
        input dimension and the last element is the output dimension
        num_groups: optional, number of groups for embedding
        embedding_dim: optional, dimension of group embeddings
        '''
        super(_general_MLP, self).__init__()
        
        # Optional group embedding layer
        self.use_embeddings = num_groups is not None and embedding_dim is not None
        if self.use_embeddings:
            self.group_embedding = nn.Embedding(num_groups, embedding_dim)
            # Adjust the first layer width to include embedding dimension
            layer_widths = layer_widths.copy()
            layer_widths[0] += embedding_dim
        
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

        
    def forward(self, x, group_memberships=None):
        # Handle group embeddings if enabled
        if self.use_embeddings:
            if group_memberships is None:
                raise ValueError("group_memberships required when using embeddings")
            
            batch_size = x.size(0)
            
            # Vectorized group embedding computation
            # Collect all group indices and their corresponding sample indices
            all_group_indices = []
            sample_indices = []
            group_counts = torch.zeros(batch_size, device=x.device)
            
            for i, sample_groups in enumerate(group_memberships):
                if len(sample_groups) > 0:
                    all_group_indices.extend(sample_groups)
                    sample_indices.extend([i] * len(sample_groups))
                    group_counts[i] = len(sample_groups)
            
            # Initialize group embeddings tensor
            group_embeds = torch.zeros(batch_size, self.group_embedding.embedding_dim, device=x.device)
            
            if len(all_group_indices) > 0:
                # Convert to tensors
                all_group_indices = torch.tensor(all_group_indices, device=x.device, dtype=torch.long)
                sample_indices = torch.tensor(sample_indices, device=x.device, dtype=torch.long)
                
                # Get all embeddings at once
                all_embeds = self.group_embedding(all_group_indices)  # size should be [len(all_group_indices), embedding_dim]
                
                # Sum embeddings per sample using scatter_add
                # Need to expand sample_indices to match embedding dimensions
                sample_indices_expanded = sample_indices.unsqueeze(1).expand(-1, self.group_embedding.embedding_dim)
                group_embeds.scatter_add_(0, sample_indices_expanded, all_embeds)
                
                # Average by dividing by group counts (avoid division by zero)
                mask = group_counts > 0  # [batch_size]
                if mask.any():
                    group_embeds[mask] = group_embeds[mask] / group_counts[mask].unsqueeze(1)
            
            # Concatenate main features with group embeddings
            x = torch.cat([x, group_embeds], dim=1)  # [batch_size, feature_dim + embedding_dim]
        
        # Forward through MLP layers
        for idx in range(len(self.layers)-1):
            x = self.layers[idx](x)
        x = self.layers[-1](x)

        return x


class MLP:
    def __init__(self, SAVE_DIR, config, from_saved=False, save_scheme='best-val-acc'):
        """
        Simple neural network class
        """
        self.config = config
        self.SAVE_DIR = SAVE_DIR
        self.from_saved = from_saved
        self.save_scheme = save_scheme

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

        # Handle group memberships if using embeddings
        if self.use_group_embeddings:
            group_memberships_train = self._get_group_memberships(len(X_train), groups_train)
            group_memberships_val = self._get_group_memberships(len(X_val), groups_val)

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
        if self.use_group_embeddings:
            # Custom dataset for group memberships
            class GroupDataset(torch.utils.data.Dataset):
                def __init__(self, X, y, group_memberships):
                    self.X = X
                    self.y = y
                    self.group_memberships = group_memberships
                
                def __len__(self):
                    return len(self.X)
                
                def __getitem__(self, idx):
                    return self.X[idx], self.y[idx], self.group_memberships[idx]
            
            # Custom collate function to handle variable-length group memberships
            def custom_collate(batch):
                X_batch, y_batch, group_batch = zip(*batch)
                X_batch = torch.stack(X_batch)
                y_batch = torch.stack(y_batch)
                # group_batch is a list of lists with variable lengths - keep as is
                return X_batch, y_batch, list(group_batch)
            
            train_dataset = GroupDataset(X_train_tensor, y_train_tensor, group_memberships_train)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, collate_fn=custom_collate)
        else:
            train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        train_loss = []
        train_accs = []

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

        # if save_scheme is 'best-val-loss', we need to keep track of the best validation loss
        if self.save_scheme == 'best-val-loss':
            best_val_loss = float('inf')
        elif self.save_scheme == 'best-val-acc':
            best_val_acc = -1
        else:
            raise ValueError('Unknown save scheme')
        
        # train loop
        for epoch in epoch_range:
            running_loss = 0.0

            if epoch in self.lr_schedule:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr_schedule[epoch]
                    current_LR = param_group['lr']

            for i, data in enumerate(train_dataloader, 0):
                # get the inputs; data is a list of [inputs, labels] or [inputs, labels, group_memberships]
                if self.use_group_embeddings:
                    inputs, labels, group_memberships = data
                else:
                    inputs, labels = data
                    group_memberships = None
                # data already on device
                
                # zero the parameter gradients
                self.optimizer.zero_grad()
                
                # forward + backward + optimize
                outputs = self.net(inputs, group_memberships)
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
                train_groups = group_memberships_train if self.use_group_embeddings else None
                y_pred = np.argmax(self.net(X_train_tensor, train_groups).detach().cpu().numpy(), axis=1)
                train_acc = accuracy_score(y_train_tensor.cpu().numpy(), y_pred)
                print('train acc', train_acc)
                train_accs.append(train_acc)

                # validate every val_eval_epoch epochs
                if epoch % self.val_eval_epoch == 0:
                    val_groups = group_memberships_val if self.use_group_embeddings else None
                    outputs = self.net(X_val_tensor, val_groups)
                    y_pred = np.argmax(outputs.detach().cpu().numpy(), axis=1)

                    # Check fraction of 1s in validation data and predictions
                    print('y_val mean pred', np.mean(y_pred))
                    print('y true mean', np.mean(y_val_tensor.cpu().numpy()))

                    val_acc = accuracy_score(y_val_tensor.cpu().numpy(), y_pred)
                    val_loss = self.criterion(outputs, y_val_tensor)
                    print('val acc', val_acc)
                    print('val loss', val_loss)

                    if self.save_scheme == 'best-val-loss':
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            torch.save(self.net.state_dict(), self.SAVE_DIR + 'model.pt')
                    elif self.save_scheme == 'best-val-acc':
                        if val_acc > best_val_acc and epoch >= self.val_save_epoch:
                            best_val_acc = val_acc
                            torch.save(self.net.state_dict(), self.SAVE_DIR + 'model.pt')
                    else:
                        raise ValueError('Unknown save scheme')

        # save config to file
        with open(self.SAVE_DIR + 'model_config.txt', 'w') as f:
            f.write(str(self.config))

        # Load the best model
        if self.save_scheme == 'best-val-acc':
            print('Best validation accuracy:', best_val_acc)
        elif self.save_scheme == 'best-val-loss':
            print('Best validation loss:', best_val_loss)
        else:
            raise ValueError('Unknown save scheme')

        self.net.load_state_dict(torch.load(self.SAVE_DIR + 'model.pt'))
    
    def predict_proba(self, X, groups=None, with_logits=False):
        X_test_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # Handle group memberships if using embeddings
        if self.use_group_embeddings:
            if groups is None:
                raise ValueError("groups required when using embeddings")
            group_memberships = self._get_group_memberships(len(X), groups)
        else:
            group_memberships = None

        # convert predictions to probabilities
        logit = self.net(X_test_tensor, group_memberships).detach().cpu()
        p = torch.nn.functional.softmax(logit, dim=1).numpy()

        if with_logits: return p, logit.numpy()
        else: return p

    def predict(self, X, groups=None):
        X_test_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # Handle group memberships if using embeddings
        if self.use_group_embeddings:
            if groups is None:
                raise ValueError("groups required when using embeddings")
            group_memberships = self._get_group_memberships(len(X), groups)
        else:
            group_memberships = None
            
        nn_preds = np.argmax(self.net(X_test_tensor, group_memberships).detach().cpu().numpy(), axis=1)

        return nn_preds
    
    # Can potentially eventually optimize this by using a tensor of group memberships
    def _get_group_memberships(self, n_samples, groups):
        """
        Convert group membership lists to per-sample group membership lists.
        Returns a list where element i contains the list of group indices that sample i belongs to.
        """
        group_memberships = [[] for _ in range(n_samples)]
        
        for group_idx, group in enumerate(groups):
            for sample_idx in group:
                group_memberships[sample_idx].append(group_idx)
            
        return group_memberships
    
    def load_net(self, arch, from_saved):
        # Pass embedding parameters if using embeddings
        if self.use_group_embeddings:
            net = _general_MLP(arch, self.num_groups, self.embedding_dim)
        else:
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
        # Handle group embeddings or features
        self.include_groups_as_features = config.get('include_groups_as_features', False)
        self.use_group_embeddings = config.get('use_group_embeddings', False)
        
        if self.include_groups_as_features and self.use_group_embeddings:
            raise ValueError('Cannot use both include_groups_as_features and use_group_embeddings')

        if self.use_group_embeddings:
            if 'num_groups' not in config:
                raise ValueError('num_groups must be specified when use_group_embeddings is True')
            if 'embedding_dim' not in config:
                raise ValueError('embedding_dim must be specified when use_group_embeddings is True')
            self.num_groups = config['num_groups']
            self.embedding_dim = config['embedding_dim']
            print(f'Using group embeddings: {self.num_groups} groups, {self.embedding_dim} dimensions')
        
        if self.include_groups_as_features:
            if 'num_groups' not in config:
                raise ValueError('num_groups must be specified when include_groups_as_features is True')
            
            # Calculate the number of groups and modify the architecture appropriately
            print('arch before: ', self.arch)
            self.arch[0] += config['num_groups']
            print('arch after: ', self.arch)

        # require momentum for SGD
        assert self.optim_name != 'sgd' or self.momentum is not None, 'Momentum must be specified for SGD optimizer'

        # sanity check
        if 0 not in self.lr_schedule:
            raise ValueError('Learning rate schedule must start at / contain epoch 0')
        if config['val_save_epoch'] > config['epochs'] - 1:
            raise ValueError(('Note: val_save_epoch must be <= (# epochs - 1); ' +
                              'model only saved when (# epochs elapsed) > val_save_epoch.'))
        
        # Determines after how many epochs we start saving the model based on val set
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

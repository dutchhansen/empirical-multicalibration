from models.SimpleModel import LogisticRegression, NaiveBayes, RandomForest
from models.SimpleModel import SVM, RandomPredictor, DecisionTree
from models.MLP import MLP
from models.LanguageResNet import LanguageResNet
from models.ImageResNet import ImageResNet
from models.DistilBert import DistilBertClassifier
from models.ViT import ViTClassifier
import numpy as np

class Model:
    def __init__(self, model_name, SAVE_DIR=None, config=None, from_saved=False, **kwargs):
        self.name = model_name
        self.SAVE_DIR = SAVE_DIR
        self.config = config
        model_found = False

        self.calib_frac = config['calib_frac']
        dataset_obj = kwargs['dataset_obj'] if 'dataset_obj' in kwargs else None
        saved_epoch = kwargs['saved_epoch'] if 'saved_epoch' in kwargs else 0
        save_scheme = kwargs['save_scheme'] if 'save_scheme' in kwargs else 'best-val-acc'

        # verify save_scheme
        assert save_scheme in ['best-val-acc', 'all-epochs']
        if self.name not in ['DistilBert', 'ViT', 'LanguageResNet']:
            err_msg = f'all-epoch saving not supported for {self.name} model.'
            assert (save_scheme == 'best-val-acc'), err_msg
        
        # simple models
        if self.name == 'LogisticRegression':
            model_found = True
            self.model = LogisticRegression(config)

        elif self.name == 'NaiveBayes':
            model_found = True
            self.model = NaiveBayes(config)

        elif self.name == 'RandomForest':
            model_found = True
            self.model = RandomForest(config)

        elif self.name == 'SVM':
            model_found = True
            self.model = SVM(config)

        elif self.name == 'RandomPredictor':
            model_found = True
            self.model = RandomPredictor(config)

        elif self.name == 'DecisionTree':
            model_found = True
            self.model = DecisionTree(config)
            
        else:
            # need to save model for evaluation
            if self.SAVE_DIR is None:
                raise ValueError('Must provide a save directory for MLP model.')

            # need to save before training ends
            if self.config['val_save_epoch'] > self.config['epochs'] - 1:
                raise ValueError(('Note: val_save_epoch must be <= (# epochs - 1); ' +
                                  'model only saved when (# epochs elapsed) > val_save_epoch.'))

        # MLP
        if self.name == 'MLP':
            model_found = True
            self.model = MLP(self.SAVE_DIR, self.config, from_saved=from_saved)

        # Resnet
        elif self.name == 'LanguageResNet':
            model_found = True
            self.model = LanguageResNet(self.SAVE_DIR, config, from_saved=from_saved, 
                                        dataset_obj=dataset_obj, save_scheme=save_scheme)
            
        elif self.name == 'ImageResNet':
            model_found = True
            self.model = ImageResNet(self.SAVE_DIR, config, from_saved=from_saved)

        # Transformer
        elif self.name == 'DistilBert':
            model_found = True
            self.model = DistilBertClassifier(self.SAVE_DIR, config, from_saved=from_saved, 
                                              saved_epoch=saved_epoch, save_scheme=save_scheme)

        elif self.name == 'ViT':
            model_found = True
            self.model = ViTClassifier(self.SAVE_DIR, config, from_saved=from_saved, 
                                       saved_epoch=saved_epoch, save_scheme=save_scheme)
        
        if not model_found:
            raise ValueError('Unknown model name')

    def train(self, X_train, y_train, groups_train, X_val, y_val, groups_val):
        self.model.train(X_train, y_train, groups_train, X_val, y_val, groups_val)
    
    def predict_proba(self, X, with_logits=False):
        '''
        Returns positive class probabilities.
        If with_logits=True, returns both probabilities of the 
        positive class, and logits for both classes.
        '''
        # if no training set, return .5
        if self.calib_frac == 1.0:
            p = np.ones((X.shape[0], 2)) * .5
            if with_logits: return p[:, 1], p
            else: return p[:, 1]
        
        # nontrivial training set
        p = self.model.predict_proba(X, with_logits)
        if with_logits: return (p[0][:,1], p[1])
        else: return p[:,1]

    def predict(self, X):
        # if no training set, select random class
        if self.calib_frac == 1.0:
            return np.random.choice([0, 1], size=X.shape[0])
        
        # nontrivial training set
        return self.model.predict(X)

    def load(self):
        if self.name == 'MLP':
            self.model.load()


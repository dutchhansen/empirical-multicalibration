from transformers import BertTokenizerFast, DistilBertTokenizerFast
import torch
import numpy as np
import pandas as pd

class LanguageDatasetWrapper(torch.utils.data.Dataset):
    '''
    Pytorch wrapper for language dataset.
    Allows easy tokenization, conversion to ids, and padding of samples.

    BERT/DistilBert tokenization code from WILDS, with some modifications:
    @inproceedings{wilds2021,
        title = {{WILDS}: A Benchmark of in-the-Wild Distribution Shifts},
        author = {Pang Wei Koh and Shiori Sagawa and Henrik Marklund and Sang Michael Xie and 
                    Marvin Zhang and Akshay Balsubramani and Weihua Hu and Michihiro Yasunaga and 
                    Richard Lanas Phillips and Irena Gao and Tony Lee and Etienne David and Ian Stavness and 
                    Wei Guo and Berton A. Earnshaw and Imran S. Haque and Sara Beery and Jure Leskovec and 
                    Anshul Kundaje and Emma Pierson and Sergey Levine and Chelsea Finn and Percy Liang},
        booktitle = {International Conference on Machine Learning (ICML)},
        year = {2021}
    }
    '''
    def __init__(self, X, y, config):
        self.X = X
        if y is None:
            self.y = np.zeros(len(X), )
        else: 
            self.y = y
        self._init_transform(config)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = self.X[idx]
        label = torch.tensor(self.y[idx])
        # transform
        x_t = self.transform(sample)

        return x_t, label

    def _init_transform(self, config):
        (
            self.tokenizer,
            self.vocab,
            self.max_token_len,
        ) = (
            config['tokenizer'] if 'tokenizer' in config else None,
            config['vocab'] if 'vocab' in config else None,
            config['max_token_len'],
        )

        model = config['model']
        if model == "ResNet":
            # check that tokenizer and vocab are provided
            assert self.tokenizer is not None
            assert self.vocab is not None

            self.transform = self.base_transform
        
        elif (model == "distilbert-base-uncased" or 
              model == "bert-base-uncased"):
            self.transform = self.init_bert_transform(config)
            
        else:
            raise ValueError(f"No transform supported for model: {model}.")

    def base_transform(self, x):
        # tokenize
        tokenized = self.tokenizer(x)[:self.max_token_len]
        # convert to ids
        ids = torch.tensor([self.vocab[t] for t in tokenized])
        # pad sequence
        padding = torch.full((self.max_token_len - len(ids), ), self.vocab['<pad>'])
        padded_ids = torch.cat([ids, padding])

        return padded_ids
    
    def init_bert_transform(self, config):
        def get_bert_tokenizer(model):
            if model == "bert-base-uncased":
                return BertTokenizerFast.from_pretrained(model)
            elif model == "distilbert-base-uncased":
                return DistilBertTokenizerFast.from_pretrained(model)
            else:
                raise ValueError(f"Model: {model} not recognized.")

        assert (config['model'] == 'bert-base-uncased' or 
                config['model'] == 'distilbert-base-uncased')

        tokenizer = get_bert_tokenizer(config['model'])

        def transform(text):
            if pd.isna(text):
                text = ""
            tokens = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=config['max_token_len'],
                return_tensors="pt",)
            
            # Bert
            if config['model'] == "bert-base-uncased":
                x = torch.stack(
                    (
                        tokens["input_ids"],
                        tokens["attention_mask"],
                        tokens["token_type_ids"],
                    ),
                    dim=2,
                )
            
            # DistilBert
            elif config['model'] == "distilbert-base-uncased":
                x = torch.stack((tokens["input_ids"], tokens["attention_mask"]), dim=2)

            # First shape dim is always 1
            x = torch.squeeze(x, dim=0)
            return x

        return transform
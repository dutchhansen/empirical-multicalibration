from torchtext.data.utils import get_tokenizer

hyperparameters = {
    'LogisticRegression': {
        'ACSIncome': {
            0: {
                'C': 0.4,
            },
            0.01: {
                'C': 1,
            },
            0.05: {
                'C': 0.4,
            },
            0.1: {
                'C': 2,
            },
            0.2: {
                'C': 2,
            },
            0.4: {
                'C': 0.4,
            },
            0.6: {
                'C': 2,
            },
            0.8: {
                'C': 0.4,
            },
            # 1.0 completely arbitrary
            1.0: {
                'C': 0.4,
            },
        },
        'BankMarketing': {
            0: {
                'C': 2,
            },
            0.01: {
                'C': 1,
            },
            0.05: {
                'C': 2,
            },
            0.1: {
                'C': 0.4,
            },
            0.2: {
                'C': 0.4,
            },
            0.4: {
                'C': 0.4,
            },
            0.6: {
                'C': 4,
            },
            0.8: {
                'C': 1,
            },
            # 1.0 completely arbitrary
            1.0: {
                'C': 1,
            },
        },
        'CreditDefault': {
            0: {
                'C': 4,
            },
            0.01: {
                'C': 4,
            },
            0.05: {
                'C': 4,
            },
            0.1: {
                'C': 4,
            },
            0.2: {
                'C': 2,
            },
            0.4: {
                'C': 4,
            },
            0.6: {
                'C': 4,
            },
            0.8: {
                'C': 4,
            },
            # 1.0 completely arbitrary
            1.0: {
                'C': 4,
            },
        },
        'HMDA': {
            0: {
                'C': 4,
            },
            0.01: {
                'C': 4,
            },
            0.05: {
                'C': 0.4,
            },
            0.1: {
                'C': 0.4,
            },
            0.2: {
                'C': 0.4,
            },
            0.4: {
                'C': 0.4,
            },
            0.6: {
                'C': 4,
            },
            0.8: {
                'C': 1,
            },
            # 1.0 completely arbitrary
            1.0: {
                'C': 1,
            },
        },
        'MEPS': {
            0: {
                'C': 0.4,
            },
            0.01: {
                'C': 2,
            },
            0.05: {
                'C': 1,
            },
            0.1: {
                'C': 0.4,
            },
            0.2: {
                'C': 2,
            },
            0.4: {
                'C': 0.4,
            },
            0.6: {
                'C': 2,
            },
            0.8: {
                'C': 0.4,
            },
            # 1.0 completely arbitrary
            1.0: {
                'C': 0.4,
            },
        }
    },
    'RandomForest': {
        'ACSIncome': {
            0: {
                'n_estimators': 100,
                'max_depth': 20,
                'min_samples_split': 10,
            },
            0.01: {
                'n_estimators': 100,
                'max_depth': 20,
                'min_samples_split': 5,
            },
            0.05: {
                'n_estimators': 100,
                'max_depth': 20,
                'min_samples_split': 10,
            },
            0.1: {
                'n_estimators': 100,
                'max_depth': 20,
                'min_samples_split': 10,
            },
            0.2: {
                'n_estimators': 100,
                'max_depth': 20,
                'min_samples_split': 10,
            },
            0.4: {
                'n_estimators': 100,
                'max_depth': 20,
                'min_samples_split': 10,
            },
            0.6: {
                'n_estimators': 100,
                'max_depth': 20,
                'min_samples_split': 10,
            },
            0.8: {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
            },
            # 1.0 completely arbitrary
            1.0: {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
            },
        },
        'BankMarketing': {
            0: {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 10,
            },
            0.01: {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 5,
            },
            0.05: {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 10,
            },
            0.1: {
                'n_estimators': 100,
                'max_depth': 20,
                'min_samples_split': 10,
            },
            0.2: {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 10,
            },
            0.4: {
                'n_estimators': 100,
                'max_depth': 20,
                'min_samples_split': 10,
            },
            0.6: {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 5,
            },
            0.8: {
                'n_estimators': 100,
                'max_depth': 20,
                'min_samples_split': 10,
            },
            # 1.0 completely arbitrary
            1.0: {
                'n_estimators': 100,
                'max_depth': 20,
                'min_samples_split': 10,
            },
        },
        'CreditDefault': {
            0: {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
            },
            0.01: {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
            },
            0.05: {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 10,
            },
            0.1: {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
            },
            0.2: {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
            },
            0.4: {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
            },
            0.6: {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 10,
            },
            0.8: {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 10,
            },
            # 1.0 completely arbitrary
            1.0: {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 10,
            },
        },
        'HMDA': {
            0: {
                'n_estimators': 100,
                'max_depth': 20,
                'min_samples_split': 5,
            },
            0.01: {
                'n_estimators': 100,
                'max_depth': 20,
                'min_samples_split': 10,
            },
            0.05: {
                'n_estimators': 100,
                'max_depth': 50,
                'min_samples_split': 10,
            },
            0.1: {
                'n_estimators': 100,
                'max_depth': 20,
                'min_samples_split': 10,
            },
            0.2: {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 10,
            },
            0.4: {
                'n_estimators': 100,
                'max_depth': 20,
                'min_samples_split': 10,
            },
            0.6: {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 10,
            },
            0.8: {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 5,
            },
            # 1.0 completely arbitrary
            1.0: {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 5,
            },
        },
        'MEPS': {
            0: {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 10,
            },
            0.01: {
                'n_estimators': 100,
                'max_depth': 20,
                'min_samples_split': 5,
            },
            0.05: {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 5,
            },
            0.1: {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 10,
            },
            0.2: {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
            },
            0.4: {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 5,
            },
            0.6: {
                'n_estimators': 100,
                'max_depth': 20,
                'min_samples_split': 10,
            },
            0.8: {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 10,
            },
            # 1.0 completely arbitrary
            1.0: {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 10,
            },
        }
    },
    'NaiveBayes': {
        'ACSIncome': {
            0: {},
            0.01: {},
            0.05: {},
            0.1: {},
            0.2: {},
            0.4: {},
            0.6: {},
            0.8: {},
            1.0: {},
        },
        'BankMarketing': {
            0: {},
            0.01: {},
            0.05: {},
            0.1: {},
            0.2: {},
            0.4: {},
            0.6: {},
            0.8: {},
            1.0: {},
        },
        'CreditDefault': {
            0: {},
            0.01: {},
            0.05: {},
            0.1: {},
            0.2: {},
            0.4: {},
            0.6: {},
            0.8: {},
            1.0: {},
        },
        'HMDA': {
            0: {},
            0.01: {},
            0.05: {},
            0.1: {},
            0.2: {},
            0.4: {},
            0.6: {},
            0.8: {},
            1.0: {},
        },
        'MEPS': {
            0: {},
            0.01: {},
            0.05: {},
            0.1: {},
            0.2: {},
            0.4: {},
            0.6: {},
            0.8: {},
            1.0: {},
        }
    },
    'SVM': {
        'ACSIncome': {
            0: {
                'alpha': 0.001,
                'max_iter': 1000,
                'scale_data': True,
            },
            0.01: {
                'alpha': 0.0001,
                'max_iter': 1000,
                'scale_data': True,
            },
            0.05: {
                'alpha': 0.0001,
                'max_iter': 1000,
                'scale_data': True,
            },
            0.1: {
                'alpha': 0.0001,
                'max_iter': 1000,
                'scale_data': True,
            },
            0.2: {
                'alpha': 0.0001,
                'max_iter': 1000,
                'scale_data': True,
            },
            0.4: {
                'alpha': 0.0001,
                'max_iter': 1000,
                'scale_data': True,
            },
            0.6: {
                'alpha': 1e-05,
                'max_iter': 1000,
                'scale_data': True,
            },
            0.8: {
                'alpha': 1e-05,
                'max_iter': 1000,
                'scale_data': True,
            },
            # 1.0 completely arbitrary
            1.0: {
                'alpha': 1e-05,
                'max_iter': 1000,
                'scale_data': True,
            },
        },
        'BankMarketing': {
            0: {
                'alpha': 0.0001,
                'max_iter': 1000,
                'scale_data': True,
            },
            0.01: {
                'alpha': 0.0001,
                'max_iter': 1000,
                'scale_data': True,
            },
            0.05: {
                'alpha': 0.0001,
                'max_iter': 1000,
                'scale_data': True,
            },
            0.1: {
                'alpha': 0.0001,
                'max_iter': 1000,
                'scale_data': True,
            },
            0.2: {
                'alpha': 0.0001,
                'max_iter': 1000,
                'scale_data': True,
            },
            0.4: {
                'alpha': 0.001,
                'max_iter': 1000,
                'scale_data': True,
            },
            0.6: {
                'alpha': 0.01,
                'max_iter': 1000,
                'scale_data': True,
            },
            0.8: {
                'alpha': 0.01,
                'max_iter': 1000,
                'scale_data': True,
            },
            # 1.0 completely arbitrary
            1.0: {
                'alpha': 0.01,
                'max_iter': 1000,
                'scale_data': True,
            },
        },
        'CreditDefault': {
            0: {
                'alpha': 0.01,
                'max_iter': 1000,
                'scale_data': True,
            },
            0.01: {
                'alpha': 0.0001,
                'max_iter': 1000,
                'scale_data': True,
            },
            0.05: {
                'alpha': 0.01,
                'max_iter': 1000,
                'scale_data': True,
            },
            0.1: {
                'alpha': 0.01,
                'max_iter': 1000,
                'scale_data': True,
            },
            0.2: {
                'alpha': 0.01,
                'max_iter': 1000,
                'scale_data': True,
            },
            0.4: {
                'alpha': 0.01,
                'max_iter': 1000,
                'scale_data': True,
            },
            0.6: {
                'alpha': 0.01,
                'max_iter': 1000,
                'scale_data': True,
            },
            0.8: {
                'alpha': 0.01,
                'max_iter': 1000,
                'scale_data': True,
            },
            # 1.0 completely arbitrary
            1.0: {
                'alpha': 0.01,
                'max_iter': 1000,
                'scale_data': True,
            },
        },
        'HMDA': {
            0: {
                'alpha': 0.0001,
                'max_iter': 1000,
                'scale_data': True,
            },
            0.01: {
                'alpha': 0.01,
                'max_iter': 1000,
                'scale_data': True,
            },
            0.05: {
                'alpha': 0.01,
                'max_iter': 1000,
                'scale_data': True,
            },
            0.1: {
                'alpha': 1e-05,
                'max_iter': 1000,
                'scale_data': True,
            },
            0.2: {
                'alpha': 1e-05,
                'max_iter': 1000,
                'scale_data': True,
            },
            0.4: {
                'alpha': 1e-05,
                'max_iter': 1000,
                'scale_data': True,
            },
            0.6: {
                'alpha': 0.0001,
                'max_iter': 1000,
                'scale_data': True,
            },
            0.8: {
                'alpha': 0.0001,
                'max_iter': 1000,
                'scale_data': True,
            },
            # 1.0 completely arbitrary
            1.0: {
                'alpha': 0.0001,
                'max_iter': 1000,
                'scale_data': True,
            },
        },
        'MEPS': {
            0: {
                'alpha': 0.01,
                'max_iter': 1000,
                'scale_data': True,
            },
            0.01: {
                'alpha': 0.0001,
                'max_iter': 1000,
                'scale_data': True,
            },
            0.05: {
                'alpha': 0.0001,
                'max_iter': 1000,
                'scale_data': True,
            },
            0.1: {
                'alpha': 0.0001,
                'max_iter': 1000,
                'scale_data': True,
            },
            0.2: {
                'alpha': 0.0001,
                'max_iter': 1000,
                'scale_data': True,
            },
            0.4: {
                'alpha': 0.0001,
                'max_iter': 1000,
                'scale_data': True,
            },
            0.6: {
                'alpha': 0.001,
                'max_iter': 1000,
                'scale_data': True,
            },
            0.8: {
                'alpha': 0.01,
                'max_iter': 1000,
                'scale_data': True,
            },
            # 1.0 completely arbitrary
            1.0: {
                'alpha': 0.01,
                'max_iter': 1000,
                'scale_data': True,
            },
        }
    },
    'DecisionTree': {
        'ACSIncome': {
            0: {
                'max_depth': 10,
                'min_samples_split': 10,
            },
            0.01: {
                'max_depth': 10,
                'min_samples_split': 10,
            },
            0.05: {
                'max_depth': 10,
                'min_samples_split': 2,
            },
            0.1: {
                'max_depth': 10,
                'min_samples_split': 10,
            },
            0.2: {
                'max_depth': 10,
                'min_samples_split': 10,
            },
            0.4: {
                'max_depth': 10,
                'min_samples_split': 5,
            },
            0.6: {
                'max_depth': 10,
                'min_samples_split': 10,
            },
            0.8: {
                'max_depth': 10,
                'min_samples_split': 10,
            },
            # 1.0 completely arbitrary
            1.0: {
                'max_depth': 10,
                'min_samples_split': 10,
            },
        },
        'BankMarketing': {
            0: {
                'max_depth': 10,
                'min_samples_split': 2,
            },
            0.01: {
                'max_depth': 10,
                'min_samples_split': 5,
            },
            0.05: {
                'max_depth': 10,
                'min_samples_split': 10,
            },
            0.1: {
                'max_depth': 10,
                'min_samples_split': 10,
            },
            0.2: {
                'max_depth': 10,
                'min_samples_split': 10,
            },
            0.4: {
                'max_depth': 10,
                'min_samples_split': 2,
            },
            0.6: {
                'max_depth': 10,
                'min_samples_split': 10,
            },
            0.8: {
                'max_depth': 10,
                'min_samples_split': 5,
            },
            # 1.0 completely arbitrary
            1.0: {
                'max_depth': 10,
                'min_samples_split': 5,
            },
        },
        'CreditDefault': {
            0: {
                'max_depth': 10,
                'min_samples_split': 2,
            },
            0.01: {
                'max_depth': 10,
                'min_samples_split': 5,
            },
            0.05: {
                'max_depth': 10,
                'min_samples_split': 10,
            },
            0.1: {
                'max_depth': 10,
                'min_samples_split': 5,
            },
            0.2: {
                'max_depth': 10,
                'min_samples_split': 10,
            },
            0.4: {
                'max_depth': 10,
                'min_samples_split': 2,
            },
            0.6: {
                'max_depth': 10,
                'min_samples_split': 5,
            },
            0.8: {
                'max_depth': 10,
                'min_samples_split': 10,
            },
            # 1.0 completely arbitrary
            1.0: {
                'max_depth': 10,
                'min_samples_split': 10,
            },
        },
        'HMDA': {
            0: {
                'max_depth': 10,
                'min_samples_split': 10,
            },
            0.01: {
                'max_depth': 10,
                'min_samples_split': 2,
            },
            0.05: {
                'max_depth': 10,
                'min_samples_split': 10,
            },
            0.1: {
                'max_depth': 10,
                'min_samples_split': 5,
            },
            0.2: {
                'max_depth': 10,
                'min_samples_split': 2,
            },
            0.4: {
                'max_depth': 10,
                'min_samples_split': 5,
            },
            0.6: {
                'max_depth': 10,
                'min_samples_split': 2,
            },
            0.8: {
                'max_depth': 10,
                'min_samples_split': 2,
            },
            # 1.0 completely arbitrary
            1.0: {
                'max_depth': 10,
                'min_samples_split': 2,
            },
        },
        'MEPS': {
            0: {
                'max_depth': 10,
                'min_samples_split': 10,
            },
            0.01: {
                'max_depth': 10,
                'min_samples_split': 10,
            },
            0.05: {
                'max_depth': 10,
                'min_samples_split': 5,
            },
            0.1: {
                'max_depth': 10,
                'min_samples_split': 10,
            },
            0.2: {
                'max_depth': 10,
                'min_samples_split': 10,
            },
            0.4: {
                'max_depth': 10,
                'min_samples_split': 10,
            },
            0.6: {
                'max_depth': 10,
                'min_samples_split': 2,
            },
            0.8: {
                'max_depth': 10,
                'min_samples_split': 5,
            },
            # 1.0 completely arbitrary
            1.0: {
                'max_depth': 10,
                'min_samples_split': 5,
            },
        }
    },
    'MLP': {
        'ACSIncome': {
            0: {
                'batch_size': 128,
                'lr_schedule': {
                    0: 0.001,
                },
                'weight_decay': 0,
                'optim': 'adam',
                'epochs': 50,
                'arch': [10, 128, 'BN', 256, 'BN', 128, 2],
                'momentum': None,
            },
            0.01: {
                'batch_size': 128,
                'lr_schedule': {
                    0: 0.001,
                },
                'weight_decay': 0,
                'optim': 'adam',
                'epochs': 50,
                'arch': [10, 128, 'BN', 256, 'BN', 128, 2],
                'momentum': None,
            },
            0.05: {
                'batch_size': 128,
                'lr_schedule': {
                    0: 0.001,
                },
                'weight_decay': 0,
                'optim': 'adam',
                'epochs': 50,
                'arch': [10, 128, 'BN', 256, 'BN', 128, 2],
                'momentum': None,
            },
            0.1: {
                'batch_size': 128,
                'lr_schedule': {
                    0: 0.001,
                },
                'weight_decay': 0,
                'optim': 'adam',
                'epochs': 50,
                'arch': [10, 128, 'BN', 256, 'BN', 128, 2],
                'momentum': None,
            },
            0.4: {
                'batch_size': 64,
                'lr_schedule': {
                    0: 0.001,
                },
                'weight_decay': 0,
                'optim': 'adam',
                'epochs': 50,
                'arch': [10, 128, 'BN', 256, 'BN', 128, 2],
                'momentum': None,
            },
            0.2: {
                'batch_size': 128,
                'lr_schedule': {
                    0: 0.001,
                },
                'weight_decay': 0,
                'optim': 'adam',
                'epochs': 50,
                'arch': [10, 128, 'BN', 256, 'BN', 128, 2],
                'momentum': None,
            },
            0.6: {
                'batch_size': 64,
                'lr_schedule': {
                    0: 0.001,
                },
                'weight_decay': 0,
                'optim': 'adam',
                'epochs': 50,
                'arch': [10, 128, 'BN', 256, 'BN', 128, 2],
                'momentum': None,
            },
            0.8: {
                'batch_size': 128,
                'lr_schedule': {
                    0: 0.001,
                },
                'weight_decay': 0,
                'optim': 'adam',
                'epochs': 50,
                'arch': [10, 128, 'BN', 256, 'BN', 128, 2],
                'momentum': None,
            },
            # 1.0 completely arbitrary
            1.0: {
                'batch_size': 128,
                'lr_schedule': {
                    0: 0.001,
                },
                'weight_decay': 0,
                'optim': 'adam',
                'epochs': 50,
                'arch': [10, 128, 'BN', 256, 'BN', 128, 2],
                'momentum': None,
            },
        },
        'BankMarketing': {
            0: {
                'batch_size': 256,
                'lr_schedule': {
                    0: 0.001,
                },
                'weight_decay': 1e-05,
                'optim': 'adam',
                'epochs': 50,
                'arch': [41, 128, 'BN', 256, 'BN', 128, 2],
                'momentum': None,
            },
            0.01: {
                'batch_size': 256,
                'lr_schedule': {
                    0: 0.001,
                },
                'weight_decay': 0,
                'optim': 'adam',
                'epochs': 50,
                'arch': [41, 128, 'BN', 256, 'BN', 128, 2],
                'momentum': None,
            },
            0.05: {
                'batch_size': 256,
                'lr_schedule': {
                    0: 0.001,
                },
                'weight_decay': 1e-05,
                'optim': 'adam',
                'epochs': 50,
                'arch': [41, 128, 'BN', 256, 'BN', 128, 2],
                'momentum': None,
            },
            0.1: {
                'batch_size': 64,
                'lr_schedule': {
                    0: 0.001,
                },
                'weight_decay': 0,
                'optim': 'adam',
                'epochs': 50,
                'arch': [41, 128, 'BN', 256, 'BN', 128, 2],
                'momentum': None,
            },
            0.2: {
                'batch_size': 64,
                'lr_schedule': {
                    0: 0.001,
                },
                'weight_decay': 0,
                'optim': 'adam',
                'epochs': 50,
                'arch': [41, 128, 'BN', 256, 'BN', 128, 2],
                'momentum': None,
            },
            0.4: {
                'batch_size': 64,
                'lr_schedule': {
                    0: 0.001,
                },
                'weight_decay': 0,
                'optim': 'adam',
                'epochs': 50,
                'arch': [41, 128, 'BN', 256, 'BN', 128, 2],
                'momentum': None,
            },
            0.6: {
                'batch_size': 64,
                'lr_schedule': {
                    0: 0.001,
                },
                'weight_decay': 1e-05,
                'optim': 'adam',
                'epochs': 50,
                'arch': [41, 128, 'BN', 256, 'BN', 128, 2],
                'momentum': None,
            },
            0.8: {
                'batch_size': 64,
                'lr_schedule': {
                    0: 0.001,
                },
                'weight_decay': 1e-05,
                'optim': 'adam',
                'epochs': 50,
                'arch': [41, 100, 2],
                'momentum': None,
            },
            # 1.0 completely arbitrary
            1.0: {
                'batch_size': 64,
                'lr_schedule': {
                    0: 0.001,
                },
                'weight_decay': 1e-05,
                'optim': 'adam',
                'epochs': 50,
                'arch': [41, 100, 2],
                'momentum': None,
            },
        },
        'CreditDefault': {
            0: {
                'batch_size': 64,
                'lr_schedule': {
                    0: 0.0001,
                },
                'weight_decay': 0,
                'optim': 'adam',
                'epochs': 5,
                'arch': [118, 128, 256, 128, 2],
                'momentum': None,
            },
            0.01: {
                'batch_size': 32,
                'lr_schedule': {
                    0: 0.0001,
                },
                'weight_decay': 0,
                'optim': 'adam',
                'epochs': 5,
                'arch': [118, 128, 256, 128, 2],
                'momentum': None,
            },
            0.05: {
                'batch_size': 64,
                'lr_schedule': {
                    0: 0.001,
                },
                'weight_decay': 0,
                'optim': 'adam',
                'epochs': 5,
                'arch': [118, 100, 2],
                'momentum': None,
            },
            0.1: {
                'batch_size': 16,
                'lr_schedule': {
                    0: 0.0001,
                },
                'weight_decay': 0,
                'optim': 'adam',
                'epochs': 5,
                'arch': [118, 128, 256, 128, 2],
                'momentum': None,
            },
            0.2: {
                'batch_size': 64,
                'lr_schedule': {
                    0: 0.001,
                },
                'weight_decay': 0,
                'optim': 'adam',
                'epochs': 5,
                'arch': [118, 100, 2],
                'momentum': None,
            },
            0.4: {
                'batch_size': 128,
                'lr_schedule': {
                    0: 0.01,
                },
                'weight_decay': 0,
                'optim': 'adam',
                'epochs': 5,
                'arch': [118, 128, 256, 128, 2],
                'momentum': None,
            },
            0.6: {
                'batch_size': 32,
                'lr_schedule': {
                    0: 0.001,
                },
                'weight_decay': 0,
                'optim': 'adam',
                'epochs': 5,
                'arch': [118, 100, 2],
                'momentum': None,
            },
            0.8: {
                'batch_size': 32,
                'lr_schedule': {
                    0: 0.001,
                },
                'weight_decay': 0,
                'optim': 'adam',
                'epochs': 5,
                'arch': [118, 100, 2],
                'momentum': None,
            },
            # 1.0 completely arbitrary
            1.0: {
                'batch_size': 32,
                'lr_schedule': {
                    0: 0.001,
                },
                'weight_decay': 0,
                'optim': 'adam',
                'epochs': 5,
                'arch': [118, 100, 2],
                'momentum': None,
            },
        },
        'HMDA': {
            0: {
                'batch_size': 128,
                'lr_schedule': {
                    0: 0.001,
                },
                'weight_decay': 0,
                'optim': 'adam',
                'epochs': 30,
                'arch': [89, 128, 'BN', 128, 2],
                'momentum': None,
            },
            0.01: {
                'batch_size': 128,
                'lr_schedule': {
                    0: 0.001,
                },
                'weight_decay': 1e-05,
                'optim': 'adam',
                'epochs': 30,
                'arch': [89, 128, 'BN', 128, 2],
                'momentum': None,
            },
            0.05: {
                'batch_size': 128,
                'lr_schedule': {
                    0: 0.001,
                },
                'weight_decay': 0,
                'optim': 'adam',
                'epochs': 30,
                'arch': [89, 100, 2],
                'momentum': None,
            },
            0.1: {
                'batch_size': 128,
                'lr_schedule': {
                    0: 0.001,
                },
                'weight_decay': 1e-05,
                'optim': 'adam',
                'epochs': 30,
                'arch': [89, 128, 'BN', 128, 2],
                'momentum': None,
            },
            0.2: {
                'batch_size': 128,
                'lr_schedule': {
                    0: 0.001,
                },
                'weight_decay': 0,
                'optim': 'adam',
                'epochs': 30,
                'arch': [89, 100, 2],
                'momentum': None,
            },
            0.4: {
                'batch_size': 128,
                'lr_schedule': {
                    0: 0.001,
                },
                'weight_decay': 1e-05,
                'optim': 'adam',
                'epochs': 30,
                'arch': [89, 100, 2],
                'momentum': None,
            },
            0.6: {
                'batch_size': 128,
                'lr_schedule': {
                    0: 0.001,
                },
                'weight_decay': 1e-05,
                'optim': 'adam',
                'epochs': 30,
                'arch': [89, 100, 2],
                'momentum': None,
            },
            0.8: {
                'batch_size': 128,
                'lr_schedule': {
                    0: 0.001,
                },
                'weight_decay': 0.0001,
                'optim': 'adam',
                'epochs': 30,
                'arch': [89, 100, 2],
                'momentum': None,
            },
            # 1.0 completely arbitrary
            1.0: {
                'batch_size': 128,
                'lr_schedule': {
                    0: 0.001,
                },
                'weight_decay': 0.0001,
                'optim': 'adam',
                'epochs': 30,
                'arch': [89, 100, 2],
                'momentum': None,
            },
        },
        'MEPS': {
            0: {
                'batch_size': 64,
                'lr_schedule': {
                    0: 0.0001,
                },
                'weight_decay': 1e-05,
                'optim': 'adam',
                'epochs': 50,
                'arch': [139, 128, 256, 128, 2],
                'momentum': None,
            },
            0.01: {
                'batch_size': 16,
                'lr_schedule': {
                    0: 0.0001,
                },
                'weight_decay': 0,
                'optim': 'adam',
                'epochs': 50,
                'arch': [139, 100, 2],
                'momentum': None,
            },
            0.05: {
                'batch_size': 64,
                'lr_schedule': {
                    0: 0.01,
                },
                'weight_decay': 0,
                'optim': 'adam',
                'epochs': 50,
                'arch': [139, 128, 256, 128, 2],
                'momentum': None,
            },
            0.1: {
                'batch_size': 32,
                'lr_schedule': {
                    0: 0.01,
                },
                'weight_decay': 0,
                'optim': 'adam',
                'epochs': 50,
                'arch': [139, 128, 256, 128, 2],
                'momentum': None,
            },
            0.2: {
                'batch_size': 32,
                'lr_schedule': {
                    0: 0.0001,
                },
                'weight_decay': 0.0001,
                'optim': 'adam',
                'epochs': 50,
                'arch': [139, 100, 2],
                'momentum': None,
            },
            0.4: {
                'batch_size': 16,
                'lr_schedule': {
                    0: 0.0001,
                },
                'weight_decay': 1e-05,
                'optim': 'adam',
                'epochs': 50,
                'arch': [139, 100, 2],
                'momentum': None,
            },
            0.6: {
                'batch_size': 32,
                'lr_schedule': {
                    0: 0.01,
                },
                'weight_decay': 1e-05,
                'optim': 'adam',
                'epochs': 50,
                'arch': [139, 128, 256, 128, 2],
                'momentum': None,
            },
            0.8: {
                'batch_size': 16,
                'lr_schedule': {
                    0: 0.01,
                },
                'weight_decay': 0.0001,
                'optim': 'adam',
                'epochs': 50,
                'arch': [139, 128, 'BN', 256, 'BN', 128, 2],
                'momentum': None
            },
            # 1.0 completely arbitrary
            1.0: {
                'batch_size': 16,
                'lr_schedule': {
                    0: 0.01,
                },
                'weight_decay': 0.0001,
                'optim': 'adam',
                'epochs': 50,
                'arch': [139, 128, 'BN', 256, 'BN', 128, 2],
                'momentum': None
            },
        }
    },
    'LanguageResNet': {
        'YelpPolarity': {
            cf: {
                'epochs': 10,
                'resnet_type': "56",
                'batch_size': 32,
                'lr_schedule': {0: 1e-4},
                'weight_decay': 0,
                'optim': 'adam',
                'nb_labels': 2,
                'in_channels': 3,
                'pretrained_embedding': 'glove',
                'embedding_dim': 512,
                'freeze_embedding': False,
                'stack_embedding': True,
                'tokenizer': get_tokenizer('basic_english'),
                'max_token_len': 70,
                'min_freq': 5,
            } for cf in [0, 0.2, 0.4]
        },
        'AmazonPolarity': {
            cf: {
                'epochs': 10,
                'resnet_type': "56",
                'batch_size': 32,
                'lr_schedule': {0: 1e-4},
                'weight_decay': 0,
                'optim': 'adam',
                'nb_labels': 2,
                'in_channels': 3,
                'pretrained_embedding': 'glove',
                'embedding_dim': 512,
                'freeze_embedding': False,
                'stack_embedding': True,
                'tokenizer': get_tokenizer('basic_english'),
                'max_token_len': 70,
                'min_freq': 5,
            } for cf in [0, 0.2, 0.4]
        },
        'CivilComments': {
            cf: {
                'epochs': 10,
                'resnet_type': "56",
                'batch_size': 32,
                'lr_schedule': {0: 1e-4},
                'weight_decay': 0,
                'optim': 'adam',
                'nb_labels': 2,
                'in_channels': 3,
                'pretrained_embedding': 'glove',
                'embedding_dim': 300,
                'freeze_embedding': False,
                'stack_embedding': True,
                'tokenizer': get_tokenizer('basic_english'),
                'max_token_len': 300,
                'min_freq': 5,
            } for cf in [0, 0.2, 0.4]
        }
    },
    'DistilBert': {
        'YelpPolarity': {
            0: {
                'max_token_len': 512,
                'epochs': 10,
                'batch_size': 16,
                'lr_schedule': {0:1e-5},
                'weight_decay': 0.01,
                'optim': 'sgd',
                'momentum': 0,
            },
            0.2: {
                'max_token_len': 512,
                'epochs': 10,
                'batch_size': 16,
                'lr_schedule': {0:1e-5},
                'weight_decay': 0.01,
                'optim': 'sgd',
                'momentum': 0,
            },
            0.4: {
                'max_token_len': 512,
                'epochs': 10,
                'batch_size': 16,
                'lr_schedule': {0:1e-5},
                'weight_decay': 0.01,
                'optim': 'sgd',
                'momentum': 0,
            },
        },
        'AmazonPolarity': {
            0: {
                'max_token_len': 512,
                'epochs': 10,
                'batch_size': 16,
                'lr_schedule': {0:1e-5},
                'weight_decay': 0.01,
                'optim': 'sgd',
                'momentum': 0,
            },
            0.2: {
                'max_token_len': 512,
                'epochs': 10,
                'batch_size': 16,
                'lr_schedule': {0:1e-5},
                'weight_decay': 0.01,
                'optim': 'sgd',
                'momentum': 0,
            },
            0.4: {
                'max_token_len': 512,
                'epochs': 10,
                'batch_size': 16,
                'lr_schedule': {0:1e-5},
                'weight_decay': 0.01,
                'optim': 'sgd',
                'momentum': 0,
            },
        },
        'CivilComments': {
            0: {
                'max_token_len': 300,
                'epochs': 10,
                'batch_size': 16,
                'lr_schedule': {0:1e-5},
                'weight_decay': 0.01,
                'optim': 'adam',
                'momentum': 0,
            },
            0.2: {
                'max_token_len': 300,
                'epochs': 10,
                'batch_size': 16,
                'lr_schedule': {0:1e-5},
                'weight_decay': 0.01,
                'optim': 'adam',
                'momentum': 0,
            },
            0.4: {
                'max_token_len': 300,
                'epochs': 10,
                'batch_size': 16,
                'lr_schedule': {0:1e-5},
                'weight_decay': 0.01,
                'optim': 'adam',
                'momentum': 0,
            },
        }
    },
    'ImageResNet': {
        'CelebA': {
            0: {
                'arch': 'resnet50',
                'pretrained': None,
                'epochs': 50,
                'batch_size': 64,
                'lr_schedule': {0: 1e-3},
                'weight_decay': 0,
                'optim': 'sgd',
                'momentum': 0.9
            },
            0.2: {
                'arch': 'resnet50',
                'pretrained': None,
                'epochs': 50,
                'batch_size': 64,
                'lr_schedule': {0: 1e-3},
                'weight_decay': 0,
                'optim': 'sgd',
                'momentum': 0.9
            },
            0.4: {
                'arch': 'resnet50',
                'pretrained': None,
                'epochs': 50,
                'batch_size': 64,
                'lr_schedule': {0: 1e-3},
                'weight_decay': 0,
                'optim': 'sgd',
                'momentum': 0.9
            },
        },
        'Camelyon17': {
            0: {
                'arch': 'densenet121',
                'pretrained': None,
                'epochs': 10,
                'batch_size': 32,
                'lr_schedule': {0: 1e-3},
                'weight_decay': 0.01,
                'optim': 'sgd',
                'momentum': 0.9
            },
            0.2: {
                'arch': 'densenet121',
                'pretrained': None,
                'epochs': 10,
                'batch_size': 32,
                'lr_schedule': {0: 1e-3},
                'weight_decay': 0.01,
                'optim': 'sgd',
                'momentum': 0.9
            },
            0.4: {
                'arch': 'densenet121',
                'pretrained': None,
                'epochs': 10,
                'batch_size': 32,
                'lr_schedule': {0: 1e-3},
                'weight_decay': 0.01,
                'optim': 'sgd',
                'momentum': 0.9
            },
        },
    },
    'ViT': {
        'CelebA': {
            cf: {
                'epochs': 10,
                'batch_size': 64,
                'lr_schedule': {0:1e-4},
                'weight_decay': 0.01,
                'optim': 'adam',
                'momentum': 0,
            } for cf in [0, 0.2, 0.4]
        },
        'Camelyon17': {
            cf: {
                'epochs': 5,
                'batch_size': 32,
                'lr_schedule': {0:1e-3},
                'weight_decay': 0.01,
                'optim': 'sgd',
                'momentum': 0.9,
            } for cf in [0, 0.2, 0.4]
        },
    },
}

def get_hyperparameters(model, dataset, calib_frac):
    return hyperparameters[model][dataset][calib_frac]
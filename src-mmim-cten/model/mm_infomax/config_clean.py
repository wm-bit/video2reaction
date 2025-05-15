args_defaults = {
    # Dropouts
    'dropout_a': 0.1,
    'dropout_v': 0.1,
    'dropout_prj': 0.1,

    # Architecture
    'n_layer': 1,
    'cpc_layers': 1,
    'd_vh': 16,
    'd_ah': 16,
    'd_vout': 16,
    'd_aout': 16,
    'd_prjh': 128,
    'pretrain_emb': 768,

    # Activations
    'mmilb_mid_activation': 'ReLU',
    'mmilb_last_activation': 'Tanh',
    'cpc_activation': 'Tanh',

    # Training
    'batch_size': 32,
    'clip': 1.0,
    'lr_main': 1e-3,
    'lr_bert': 5e-5,
    'lr_mmilb': 1e-3,
    'alpha': 0.1,
    'beta': 0.1,
    'weight_decay_main': 1e-4,
    'weight_decay_bert': 1e-4,
    'weight_decay_club': 1e-4,
}
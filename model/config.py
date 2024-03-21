# import sys
from datetime import datetime
from src.data_encoding import categ_to_resnames


config_data = {
    'dataset_filepath': "/blue/yanjun.li/riddhishthakare/PeSTo/data/datasets/RNA-SMOL/contacts_rr5A_64nn_8192_wat_train60.h5",
    'test18_filepath': "/blue/yanjun.li/riddhishthakare/PeSTo/data/datasets/RNA-SMOL/contacts_rr5A_64nn_8192_wat_test18.h5",
    'test9_filepath': "/blue/yanjun.li/riddhishthakare/PeSTo/data/datasets/RNA-SMOL/contacts_rr5A_64nn_8192_wat_test9.h5",
    # 'dataset_filepath': "/tmp/"+sys.argv[-1]+"/contacts_rr5A_64nn_8192.h5",
    'train_selection_filepath': "/blue/yanjun.li/riddhishthakare/PeSTo/datasets/RNA-SMOL/subunits_train60.txt",
    'test_selection_filepath': "/blue/yanjun.li/riddhishthakare/PeSTo/datasets/RNA-SMOL/subunits_test18.txt",
    'test9_selection_filepath': "/blue/yanjun.li/riddhishthakare/PeSTo/datasets/RNA-SMOL/subunits_test9.txt",

    'max_ba': 1,
    'max_size': 1024*8,
    'min_num_res': 32,
    'l_types': categ_to_resnames['rna'],
    'r_types': [
        categ_to_resnames['protein'],
        categ_to_resnames['dna']+categ_to_resnames['rna'],
        categ_to_resnames['ion'],
        categ_to_resnames['ligand'],
        categ_to_resnames['lipid'],
    ],
    # 'r_types': [[c] for c in categ_to_resnames['protein']],
}

config_model = {
    "em": {'N0': 30, 'N1': 32},
    "sum": [
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 8},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 8},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 8},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 8},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 8},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 8},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 8},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 8},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 16},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 16},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 16},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 16},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 16},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 16},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 16},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 16},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 32},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 32},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 32},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 32},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 32},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 32},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 32},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 32},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 64},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 64},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 64},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 64},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 64},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 64},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 64},
        {'Ns': 32, 'Nh': 2, 'Nk':3, 'nn': 64},
    ],
    "spl": {'N0': 32, 'N1': 32, 'Nh': 4},
    "dm": {'N0': 32, 'N1': 32, 'N2': 5}
}

# define run name tag
tag = datetime.now().strftime("_%Y-%m-%d_%H-%M")

config_runtime = {
    'run_name': 'i_v4_1'+tag,
    'output_dir': 'save',
    'reload': True,
    'device': 'cuda',
    'num_epochs': 40,
    'batch_size': 1,
    'log_step': 1024,
    'eval_step': 1024*8,
    'learning_rate': 1e-4,
    'pos_weight_factor': 0.55,
    'comment': "",
}

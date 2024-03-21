import os
import sys
import h5py
import json
import numpy as np
import torch as pt
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
from sklearn import metrics

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = 14

from theme import colors

from src.dataset import StructuresDataset, collate_batch_features, select_by_sid, select_by_max_ba, select_by_interface_types
from src.data_encoding import encode_structure, encode_features, extract_topology, categ_to_resnames, resname_to_categ, extract_all_contacts
from src.structure import data_to_structure, encode_bfactor, concatenate_chains
from src.structure_io import save_pdb, read_pdb
from src.scoring import bc_scoring, bc_score_names, nanmean

from collections import defaultdict

save_path = "model/save/Final_set/3_14_4"  # 91
model_filepath = os.path.join(save_path, 'model_ckpt.pt')

# add module to path
if save_path not in sys.path:
    sys.path.insert(0, save_path)
    
# load functions
from model.config import config_model, config_data
from model.data_handler import Dataset
from model.model import Model

# define device
device = pt.device("cuda")

# create model
model = Model(config_model)

# reload model
model.load_state_dict(pt.load(model_filepath, map_location=pt.device("cpu")))

# set model to inference
model = model.eval().to(device)

t9_labels = {
    '94': [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
'71': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
'78': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,],
'20':[0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
'23': [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
'29': [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
'27': [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
'77': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
'38': [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
}
def setup_dataset(config_data, r_types_sel):
    # set up dataset
    dataset = Dataset("data/datasets/contacts_rr5A_64nn_8192_wat_test9.h5")
    # selected structures
    sids_sel = np.genfromtxt("data/datasets/subunits_test9.txt", dtype=np.dtype('U'))

    # filter dataset
    m = select_by_sid(dataset, sids_sel) # select by sids
    
    # data selection criteria
    m &= select_by_max_ba(dataset, config_data['max_ba'])  # select by max assembly count

    m &= select_by_interface_types(dataset, categ_to_resnames['rna'], np.concatenate(r_types_sel))  # select by interface type

    # update dataset selection
    dataset.update_mask(m)

    # set dataset types
    dataset.set_types(categ_to_resnames['rna'], config_data['r_types'])

    # debug print
    return dataset

def eval_model(model, dataset, ids):
    p_l, y_l = [], []
    with pt.no_grad():
        for i in tqdm(ids):
            # get data
            X, ids_topk, q, M, y = dataset[i]
            # y = t9_labels[str(len(y))]
            # pack data and setup sink (IMPORTANT)
            X, ids_topk, q, M = collate_batch_features([[X, ids_topk, q, M]])

            # run model
            z = model(X.to(device), ids_topk.to(device), q.to(device), M.float().to(device))

            # prediction
            p = pt.sigmoid(z)
            # categorical predictions
            pc = pt.cat([1.0 - pt.max(p, axis=1)[0].unsqueeze(1), p], axis=1).cpu()
            yc = pt.cat([1.0 - pt.any(y > 0.5, axis=1).float().unsqueeze(1), y], axis=1).cpu()
            print("PC shape - ", pc.shape)
            # data
            p_l.append(pc)
            y_l.append(yc)
            # print(yc)

    return p_l, y_l


p_l, y_l = [], []

for i in range(len(config_data['r_types'])):
    # debug print

    if i==0 or i==1 or i==2 or i==4:
        continue
    
    print(config_data['r_types'][i])

    # load datasets
    dataset = setup_dataset(config_data, [config_data['r_types'][i]])
    print("dataset: {}".format(len(dataset)))

    # parameters
    N = min(len(dataset), 512)

    # run negative examples
    ids = np.arange(len(dataset))
    np.random.shuffle(ids)
    pi_l, yi_l = eval_model(model, dataset, ids[:N])
    
    # store evaluation results
    p_l.append(pi_l)
    y_l.append(yi_l)


class_names = ["ligand" ]

# compute scores per class
scores = []
for i in range(len(y_l)):
    # extract class
    p = pt.cat(p_l[i], axis=0)[:,4]
    y = pt.cat(y_l[i], axis=0)[:,4]

    # compute scores
    s = bc_scoring(y.unsqueeze(1), p.unsqueeze(1)).squeeze().numpy()
    
    # compute F1 score
    f1 = metrics.f1_score(y.numpy().astype(int), p.numpy().round())
    
    # compute ratio of positives
    r = pt.mean(y)
    
    # store results
    scores.append(np.concatenate([s, [f1, r]]))


    # # compute roc and roc auc
    # pre, rec, _ = metrics.precision_recall_curve(y.numpy(), p.numpy())
    # auc = metrics.auc(rec, pre)
    
    # # update plot
    # plt.plot(rec, pre,  '-', color=colors[i], label="{} (auc: {:.2f})".format(class_names[i], auc))
    
# pack data
scores = np.stack(scores).T

# make table
df = pd.DataFrame(data=np.round(scores,2), index=bc_score_names+['F1', 'r'], columns=class_names)

# save dataframe
df.to_csv("type_interface_search_scores.csv")

print(df)

# # parameters
# class_names = ["ion","ligand"]

# # plot
# plt.figure(figsize=(5,4.5))
# for i in range(len(y_l)):

#     # get labels and predictions for class
#     yi = pt.cat(y_l[i], axis=0)[:,i+3]
#     pi = pt.cat(p_l[i], axis=0)[:,i+3]

#     # compute roc and roc auc
#     pre, rec, _ = metrics.precision_recall_curve(yi.numpy(), pi.numpy())
#     auc = metrics.auc(rec, pre)
    
#     # update plot
#     plt.plot(rec, pre)
    
# plt.xlim(0.0, 1.0)
# plt.ylim(0.0, 1.0)
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.savefig("results/type_interface_search_pr_auc.png")
# plt.show()
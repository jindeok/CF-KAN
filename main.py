
import argparse
from ast import parse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.DNN import *

import evaluate_utils
import data_utils
from copy import deepcopy

import random
random_seed = 123 
torch.manual_seed(random_seed) 
torch.cuda.manual_seed(random_seed) 
np.random.seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic=True 
def worker_init_fn(worker_id):
    np.random.seed(random_seed + worker_id)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ML-1M', help='choose the dataset ML-1M/Yelp/Anime')
parser.add_argument('--data_path', type=str, default='datasets/', help='load data path')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.3)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=1000, help='upper epoch limit')
parser.add_argument('--eval_epochs', type=int, default=3, help='eval per epoch')
parser.add_argument('--topN', type=str, default='[10, 20]')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--gpu', type=str, default='0', help='gpu card ID')
parser.add_argument('--save_path', type=str, default='../saved_models/', help='save model path')
parser.add_argument('--log_name', type=str, default='log', help='the log name')
parser.add_argument('--tst_w_val', default=False, help='True or False')
parser.add_argument('--round', type=int, default=1, help='record the experiment')

parser.add_argument('--dims', type=str, default='[512]', help='the dims for the DNN')
parser.add_argument('--grid_size', type=int, default=2, help='grid size for KAN model')
parser.add_argument('--dropout_rate', type=float, default=0.2, help='dropout rate for the model')

parser.add_argument('--verbose', type=int, default=1, help='verbosity level')
parser.add_argument('--save_model', type=bool, default=False, help='whether to save the model')


args = parser.parse_args()
print("args:", args)
device = torch.device("cuda" )
print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

# Load data
train_path = args.data_path + args.dataset + '/train_list.npy'
valid_path = args.data_path + args.dataset + '/valid_list.npy'
test_path = args.data_path + args.dataset + '/test_list.npy'

train_data, valid_y_data, test_y_data, n_user, n_item = data_utils.data_load(train_path, valid_path, test_path)
train_dataset = data_utils.Data(torch.FloatTensor(train_data.A))
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)
test_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

if args.tst_w_val:
    tv_dataset = data_utils.Data(torch.FloatTensor(train_data.A) + torch.FloatTensor(valid_y_data.A))
    test_twv_loader = DataLoader(tv_dataset, batch_size=args.batch_size, shuffle=False)
mask_tv = train_data + valid_y_data
print('data ready.')


# Load model
hidden_dims = eval(args.dims)
model = KANAutoencoder(n_item, hidden_dims, args.grid_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
print("models ready.")

def evaluate(data_loader, data_te, mask_his, topN):
    model.eval()
    e_idxlist = list(range(mask_his.shape[0]))
    e_N = mask_his.shape[0]

    predict_items = []
    target_items = []
    for i in range(e_N):
        target_items.append(data_te[i, :].nonzero()[1].tolist())
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            his_data = mask_his[e_idxlist[batch_idx*args.batch_size:batch_idx*args.batch_size+len(batch)]]
            batch = batch.to(device)
            prediction = model(batch)
            prediction[his_data.nonzero()] = -np.inf
            _, indices = torch.topk(prediction, topN[-1])
            indices = indices.cpu().numpy().tolist()
            predict_items.extend(indices)

    test_results = evaluate_utils.computeTopNAccuracy(target_items, predict_items, topN)
    return test_results

# Training loop
best_recall, best_epoch = -100, 0
best_test_result = None
print("Start training...")
for epoch in range(1, args.epochs + 1):
    if epoch - best_epoch >= 20:
        print('-'*18)
        print('Exiting from training early')
        break

    model.train()
    start_time = time.time()
    batch_count = 0
    total_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)
        batch_count += 1
        optimizer.zero_grad()
        outputs = model(batch)
        #loss = nn.CrossEntropyLoss()(outputs, batch) + 0.3 * model.regularization_loss() # for Yelp
        loss = nn.MSELoss()(outputs, batch) + 0.3 * model.regularization_loss()
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    if epoch % args.eval_epochs == 0:
        valid_results = evaluate(test_loader, valid_y_data, train_data, eval(args.topN))
        if args.tst_w_val:
            test_results = evaluate(test_twv_loader, test_y_data, mask_tv, eval(args.topN))
        else:
            test_results = evaluate(test_loader, test_y_data, mask_tv, eval(args.topN))
        evaluate_utils.print_results(None, valid_results, test_results)

        if valid_results[1][1] > best_recall:
            best_recall, best_epoch = valid_results[1][1], epoch
            best_results = valid_results
            best_test_results = test_results
    
    print("Running Epoch {:03d} ".format(epoch) + 'train loss {:.4f}'.format(total_loss) + " costs " + time.strftime("%H: %M: %S", time.gmtime(time.time()-start_time)))
    print('---'*18)

print('==='*18)
print("End. Best Epoch {:03d} ".format(best_epoch))
evaluate_utils.print_results(None, best_results, best_test_results)   
print("End time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))





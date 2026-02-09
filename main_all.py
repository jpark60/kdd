import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pickle
import gc

import config
import utils_exemplar
import utils_icarl
import utils_data
import utils_eval
from utils_transformer import EmberTransformer

device = config.device
batch_size = config.batch_size
nb_cl_first = config.nb_cl_first
nb_cl = config.nb_cl
nb_groups = config.nb_groups
nb_total = config.nb_total
epochs = config.epochs
lr = config.lr
wght_decay = config.wght_decay
use_weight_decay_in_exemplar = config.use_weight_decay_in_exemplar
lr_patience = config.lr_patience
stop_patience = config.stop_patience
stop_floor_ep = config.stop_floor_ep
factor = config.factor
min_lr = config.min_lr
nb_cluster = config.nb_cluster
exemplar_selection = config.exemplar_selection

data_path = config.data_path
x_path = config.x_path
y_path = config.y_path
x_path_valid = config.x_path_valid
y_path_valid = config.y_path_valid
save_path = config.save_path

files_protoset = []
iteration_results = []

total_classes = nb_cl_first + (nb_groups * nb_cl)  # 22 + (4 * 5) = 42
for _ in range(total_classes):
    files_protoset.append([])

print("Mixing the classes and putting them in batches of classes...")
# torch.manual_seed(config.SEED)
# np.random.seed(config.SEED)

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

order = np.arange(total_classes)
mixing = np.arange(total_classes)
np.random.shuffle(mixing)

print(f"Class mixing: {mixing}")

print("Creating a training set ...") 
files_train = utils_data.prepare_files(x_path, y_path, mixing, order, nb_groups, nb_cl, nb_cl_first) 
files_valid = utils_data.prepare_files(x_path_valid, y_path_valid, mixing, order, nb_groups, nb_cl, nb_cl_first) 

### Save the mixing and order ###
with open(f"{nb_cl}mixing.pickle", 'wb') as fp:
    pickle.dump(mixing, fp)

with open(f"{nb_cl}settings_mlp.pickle", 'wb') as fp:
    pickle.dump(order, fp)
    pickle.dump(files_train, fp)

##### ------------- Main Algorithm START -------------#####
for itera in range(nb_groups + 1):
    print(f'Batch of classes number {itera+1} arrives ...')
    
    if itera == 0:
        cur_nb_cl = nb_cl_first
        idx_iter = files_train[itera]
    else:
        cur_nb_cl = nb_cl
        idx_iter = files_train[itera][:]
        
        total_cl_now = nb_cl_first + ((itera-1) * nb_cl)
        nb_protos_cl = int(np.ceil(nb_total * 1.0 / total_cl_now))

        for i in range(nb_cl_first + (itera-1)*nb_cl): 
            tmp_var = files_protoset[i]
            selected_exemplars = tmp_var[0:min(len(tmp_var),nb_protos_cl)]
            idx_iter += selected_exemplars

    print(f'Task {itera + 1}: Training {cur_nb_cl} classes...') 

    X_train, y_train = utils_data.read_data(x_path, y_path, mixing, idx_iter)
    X_val, y_val = utils_data.read_data(x_path_valid, y_path_valid, mixing, files_valid[itera])
    
    # DataLoader
    train_dataset = utils_data.CustomDataset(X_train, y_train)
    val_dataset = utils_data.CustomDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    if itera == 0: 
        model_train, model_test = utils_icarl.prepare_networks(
            total_classes, nb_cl_first, nb_cl, nb_groups, itera=itera, save_path=save_path, device=device
        )
        optimizer = optim.Adam(model_train.parameters(), lr=lr, weight_decay=wght_decay)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=factor, patience=lr_patience, min_lr=min_lr
        )
        def train_step(model, x, y, optimizer, loss_fn):
            model.train()
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            return loss
    else:
        model_train, model_test = utils_icarl.prepare_networks(
            total_classes, nb_cl_first, nb_cl, nb_groups, itera=itera, save_path=save_path, device=device
        )
        optimizer = optim.Adam(model_train.parameters(), lr=lr, weight_decay=wght_decay)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        def train_step(model, x, y, optimizer, loss_fn):
            model.train()
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            return loss

    current_epoch = 0
    stopped_early = False
    best_val_loss = float('inf')
    early_stop_patience_counter = 0
    best_weights = None
    
    ### Training loop ###
    for epoch in range(epochs):
        epoch_losses = []

        for step, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            loss_value = train_step(model_train, x_batch, y_batch, optimizer, loss_fn)
            epoch_losses.append(loss_value.item())

            # Print average loss every 100 batches
            if (step + 1) % 50 == 0:
                print('\r', epoch, 'epoch', step, 'batch', f'Average loss: {np.mean(epoch_losses[-100:]):.6f}', end='')
        model_train.eval()
        val_losses = []
        correct = 0
        total = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                logits, _ = model_train(x_val)
                val_loss = loss_fn(logits, y_val)
                val_losses.append(val_loss.item())
                end_class = nb_cl_first if itera == 0 else (nb_cl_first + nb_cl * itera)
                logits_limited = logits[:, :end_class]
                _, predicted = torch.max(logits_limited.data, 1)
                
                total += y_val.size(0)
                correct += (predicted == y_val).sum().item()
        val_acc = correct / total if total > 0 else 0.0
        avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
        
        improved = avg_val_loss < best_val_loss - 1e-6
        if improved:
            best_val_loss = avg_val_loss
            best_weights = model_train.state_dict().copy()
            early_stop_patience_counter = 0
        else:
            early_stop_patience_counter += 1
        
        if itera == 0:
            scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f", Val_Loss: {avg_val_loss:.4f}, Val_Acc: {val_acc:.4f}, LR: {current_lr:.6e}")
        
        if itera > 0:
            if early_stop_patience_counter >= stop_patience and epoch >= stop_floor_ep:
                stopped_early = True
                current_epoch = epoch + 1
                print(f"[EarlyStop] epoch {current_epoch}, best_val_loss {best_val_loss:.4f}")
                model_train.load_state_dict(best_weights)
                break

    print(f"[Iter {itera}] training completed")

    if not stopped_early:
        current_epoch = epochs
    # print(f"[Iter {itera}] used_train_epochs={current_epoch}")

    torch.save(model_train.state_dict(), save_path + f'model-iteration{nb_cl}-{itera}.pt')
    
    # Copy model for distillation (before exemplar refinement)
    model_before = EmberTransformer(**config.model_config).to(device)
    model_before.load_state_dict(model_train.state_dict())
    
    X_test = np.load(data_path+'/X_test.npy')
    y_test = np.load(data_path+'/y_test.npy')
    y_test = np.array([mixing[label] for label in y_test])
    end_class = nb_cl_first if itera == 0 else (nb_cl_first + nb_cl * itera)
 
    # Exemplar management
    total_cl_now = nb_cl_first + (itera * nb_cl)
    nb_protos_cl = int(np.ceil(nb_total * 1.0 / total_cl_now))
    idx_iter = files_train[itera]
    
    dataset, model = utils_icarl.reading_data_and_preparing_network(
        idx_iter, itera, batch_size, x_path, y_path, mixing, nb_groups, nb_cl, nb_cl_first, save_path, device
    )

    # Load the training samples of the current batch of classes in the feature space
    Dtot, label_dico = utils_icarl.load_class_in_feature_space(
        idx_iter, batch_size, dataset, model, mixing, device
    )

    print('Exemplar selecting...')
    if itera == 0:
        start_idx = 0
        end_idx = nb_cl_first
    else:
        start_idx = nb_cl_first + (itera-1)*nb_cl
        end_idx = nb_cl_first + itera*nb_cl

    for iter_dico in range(end_idx - start_idx):
        current_cl = start_idx + iter_dico
        ind_cl = np.where(label_dico == order[current_cl])[0]
        D = Dtot[:, ind_cl]

        if len(ind_cl) == 0:
            print(f"Warning: No samples found for class {current_cl} in current task data")
            continue

        idx_iter_arr = np.array(idx_iter)
        files_iter = idx_iter_arr[ind_cl]

        # Select exemplars based on configured method
        if exemplar_selection == 'kmeans':
            selected_exemplars_files = utils_exemplar.kmeans_exemplar_selection(nb_cluster, D, files_iter, nb_protos_cl)
        elif exemplar_selection in ['random', 'mean']:
            selected_exemplars_files = utils_exemplar.mean_exemplar_selection(D, files_iter, nb_protos_cl, method=exemplar_selection)
        else:
            raise ValueError(f"Unknown exemplar selection method: {exemplar_selection}")

        for exemplar_file in selected_exemplars_files:
            if exemplar_file not in files_protoset[current_cl]:
                files_protoset[current_cl].append(exemplar_file)

        gc.collect()
            
    with open(f"{nb_cl}_files_protoset.pickle", "wb") as fp:
        pickle.dump(files_protoset, fp)

    if itera == 0:
        # Evaluate task 1 after training
        model_test.load_state_dict(model_train.state_dict())
        eval_result_after = utils_eval.evaluate_model_performance(
            model_test, X_test, y_test, nb_cl_first, nb_cl, itera, 
            end_class
        )
        
        iteration_result_after = {
            'iteration': itera,
            'phase': 'final',
            **eval_result_after
        }
        iteration_results.append(iteration_result_after)
    
    if itera > 0:
        start_idx = nb_cl_first + (itera-1)*nb_cl
        
        all_exemplar_indices = []
        for class_id, exemplar_list in enumerate(files_protoset):
            if len(exemplar_list) > 0:
                if class_id < (nb_cl_first + itera * nb_cl):
                    selected_exemplars = exemplar_list[:min(len(exemplar_list), nb_protos_cl)]
                    all_exemplar_indices.extend(selected_exemplars)
                    print(f"Class {class_id}: Using {len(selected_exemplars)}/{len(exemplar_list)} exemplars for refinement")

        ex_lr = lr * 0.1
        ex_opt = optim.Adam(
            model_train.parameters(), 
            lr=ex_lr,
            weight_decay=wght_decay if use_weight_decay_in_exemplar else 0.0
        )

        if len(all_exemplar_indices) > 0:
            X_ex, y_ex = utils_data.read_data(x_path, y_path, mixing, all_exemplar_indices)
            
            ex_dataset = utils_data.CustomDataset(X_ex, y_ex)
            ex_loader = DataLoader(ex_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
            
            fixed_loader = ex_loader
            early_stop_config = {
                'enable': config.early_stop_enable,
                'eps_improve': config.early_stop_eps,
                'k_patience': config.early_stop_patience,
                'max_samples': config.early_stop_max_samples,
                'min_epochs': config.early_stop_min_epochs
            }
            
            max_epochs = 100
            utils_exemplar.run_exemplar_fine_tuning(
                model_train, model_before, model_test, ex_loader,
                max_epochs, loss_fn, ex_opt, start_idx, device, a=1.0, b=1.0,
                fixed_loader=fixed_loader,
                early_stop_config=early_stop_config
            )
        else:
            print('\n[Refinement] (no exemplars yet skipping)')

        torch.save(model_train.state_dict(), save_path + f'model-iteration{nb_cl}-{itera}.pt')

        print("[Refinement] done.")
        model_test.load_state_dict(model_train.state_dict())

        eval_result_after = utils_eval.evaluate_model_performance(
            model_test, X_test, y_test, nb_cl_first, nb_cl, itera, 
            end_class
        )
        
        iteration_result_after = {
            'iteration': itera,
            'phase': 'final',
            **eval_result_after
        }
        iteration_results.append(iteration_result_after)

    torch.cuda.empty_cache()
    gc.collect()

utils_eval.print_results_summary(iteration_results, nb_groups)

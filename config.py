import os
import torch
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='TraMEL')
    
    # Dataset Selection
    parser.add_argument('--dataset', type=str, choices=['cic', 'iot'], default='cic',
                        help='Dataset to use: cic (CICAndMal2017) or iot (IoT23) (default: cic)')

    # GPU and System Settings
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU device (default: 0)')
    parser.add_argument('--seed', type=int, default=93,
                        help='Random seed (default: 93)')
    
    # Training Hyperparameters
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size (default: 256)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (auto-set based on dataset)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--weight_decay', type=float, default=0.00001,
                        help='Weight decay (default: 0.00001)')
    parser.add_argument('--use_weight_decay_in_exemplar', action='store_true',
                        help='Use weight decay in exemplar training')
    
    # Continual Learning Settings (will be overridden based on dataset)
    parser.add_argument('--nb_cl_first', type=int, default=None,
                        help='Number of classes in first task (auto-set based on dataset)')
    parser.add_argument('--nb_cl', type=int, default=None,
                        help='Number of classes per subsequent task (auto-set based on dataset)')
    parser.add_argument('--nb_groups', type=int, default=None,
                        help='Number of subsequent tasks (auto-set based on dataset)')
    parser.add_argument('--nb_total', type=int, default=None,
                        help='Total exemplar budget (auto-set based on dataset)')
    
    # Learning Rate Scheduler
    parser.add_argument('--lr_patience', type=int, default=2,
                        help='LR scheduler patience (default: 2)')
    parser.add_argument('--stop_patience', type=int, default=5,
                        help='Early stopping patience (default: 5)')
    parser.add_argument('--lr_factor', type=float, default=0.5,
                        help='LR reduction factor (default: 0.5)')
    parser.add_argument('--min_lr', type=float, default=1e-7,
                        help='Minimum learning rate (default: 1e-7)')
    parser.add_argument('--stop_floor_ep', type=int, default=10,
                        help='Minimum epoch for early stopping (default: 10)')
    
    # Exemplar Training Settings
    parser.add_argument('--nb_cluster', type=int, default=None,    
                        help='K-means cluster size for exemplar selection (auto-set based on dataset)')
    parser.add_argument('--exemplar_selection', type=str, default='kmeans', 
                        choices=['kmeans', 'random', 'mean'],
                        help='Exemplar selection method: kmeans, random, or mean (default: kmeans)')
    
    # Early Stopping for Refinement (margin stopping)
    parser.add_argument('--early_stop_enable', action='store_true',
                        help='Enable early stopping for exemplar refinement')
    parser.add_argument('--early_stop_patience', type=int, default=3,
                        help='Early stopping patience (default: 3)')
    parser.add_argument('--early_stop_eps', type=float, default=1e-4,
                        help='Early stopping improvement threshold (default: 1e-4)')
    parser.add_argument('--early_stop_max_samples', type=int, default=512,
                        help='Max samples for early stopping margin computation (default: 512)')
    parser.add_argument('--early_stop_min_epochs', type=int, default=5,
                        help='Minimum epochs before early stopping can trigger (default: 5)')
    
    # Data Paths (will be overridden based on dataset)
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to data directory (auto-set based on dataset)')
    parser.add_argument('--save_path', type=str, default='./',
                        help='Path to save models (default: ./)')
    
    # Model Architecture (will be overridden based on dataset)
    parser.add_argument('--in_feats', type=int, default=None,
                        help='Input features (auto-set based on dataset)')
    parser.add_argument('--hidden', type=int, default=None,
                        help='Hidden dimension (auto-set based on dataset)')
    parser.add_argument('--num_layers', type=int, default=None,
                        help='Number of transformer layers (auto-set based on dataset)')
    parser.add_argument('--nhead', type=int, default=None,
                        help='Number of attention heads (auto-set based on dataset)')
    parser.add_argument('--dropout', type=float, default=None,
                        help='Dropout rate (auto-set based on dataset)')
    parser.add_argument('--use_cls_token', type=bool, default=None,
                        help='Use CLS token in transformer (auto-set based on dataset)')
    
    return parser.parse_args()

def get_dataset_config(dataset_name):
    """Get dataset-specific configuration"""
    if dataset_name == 'cic':
        return {
            'nb_cl_first': 22,
            'nb_cl': 5,
            'nb_groups': 4,
            'nb_total': 33000,
            'data_path': '/scratch/Malware/CIC/data/',
            'in_feats': 85,
            'epochs': 50,
            # Model architecture for cic dataset
            'hidden': 384,
            'num_layers': 6,
            'nhead': 8,
            'dropout': 0.1,
            'use_cls_token': True,
            'nb_cluster': 800,
        }
    elif dataset_name == 'iot':
        return {
            'nb_cl_first': 5,
            'nb_cl': 1,
            'nb_groups': 4,
            'nb_total': 10000,
            'data_path':'/scratch/Malware/iot23/data/',
            'in_feats': 23,
            'epochs': 40,
            'hidden': 16,
            'num_layers': 1,
            'nhead': 2,
            'dropout': 0.2,
            'use_cls_token': True,
            'nb_cluster': 600,
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

# Parse arguments and set up configuration
args = parse_arguments()

# Get dataset-specific configuration
dataset_config = get_dataset_config(args.dataset)

# Override arguments with dataset-specific values if not explicitly set
for key, value in dataset_config.items():
    if getattr(args, key) is None:
        setattr(args, key, value)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = args.seed
batch_size = args.batch_size
nb_cl_first = args.nb_cl_first
nb_cl = args.nb_cl
nb_groups = args.nb_groups
nb_total = args.nb_total
epochs = args.epochs
lr = args.lr
wght_decay = args.weight_decay
use_weight_decay_in_exemplar = args.use_weight_decay_in_exemplar
total_classes = nb_cl_first + (nb_groups * nb_cl)

lr_patience = args.lr_patience
stop_patience = args.stop_patience
stop_floor_ep = args.stop_floor_ep
factor = args.lr_factor
min_lr = args.min_lr

nb_cluster = args.nb_cluster
exemplar_selection = args.exemplar_selection

early_stop_enable = args.early_stop_enable
early_stop_patience = args.early_stop_patience
early_stop_eps = args.early_stop_eps
early_stop_max_samples = args.early_stop_max_samples
early_stop_min_epochs = args.early_stop_min_epochs

data_path = args.data_path
x_path = f"{data_path}/X_train.npy"
y_path = f"{data_path}/y_train.npy"
x_path_valid = f"{data_path}/X_valid.npy"
y_path_valid = f"{data_path}/y_valid.npy"
save_path = args.save_path
model_config = {
    'in_feats': args.in_feats,
    'num_classes': total_classes,
    'hidden': args.hidden,
    'mlp_hidden': args.hidden * 3,
    'num_layers': args.num_layers,
    'nhead': args.nhead,
    'dropout': args.dropout,
    'use_cls_token': args.use_cls_token
}
print(f"\n{'='*60}")
print(f"CONFIGURATION - Dataset: {args.dataset.upper()}")
print(f"{'='*60}")
print(f"Data Path: {data_path}")
print(f"Exemplars: Total Buffer={nb_total}")
print(f"Model: Features={args.in_feats}, Hidden={args.hidden}, Layers={args.num_layers}")
print(f"Training: Epochs={epochs}, Batch Size={batch_size}, LR={lr}")
print(f"Exemplar Selection: {exemplar_selection.upper()}")
print(f"Early Stopping: {'ENABLED' if early_stop_enable else 'DISABLED'}")
if early_stop_enable:
    print(f"  Patience={early_stop_patience}, Eps={early_stop_eps}, MaxSamples={early_stop_max_samples}, MinEpochs={early_stop_min_epochs}")
print(f"{'='*60}\n")


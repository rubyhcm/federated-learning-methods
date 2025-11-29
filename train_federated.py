"""
Train federated learning models using FedAvg, FedAvgM, and FedOpt on MNIST and CIFAR10.
Supports both IID (evenly distributed) and Non-IID (imbalanced) data distributions.

Usage:
    python train_federated.py --dataset mnist --distribution iid --method all
    python train_federated.py --dataset cifar10 --distribution non-iid --method fedopt
"""
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.models.unified_cnn import UnifiedCNN
from src.data.data_loader import load_dataset, create_federated_data, get_data_loaders
from src.algorithms.fedavg import FedAvg
from src.algorithms.fedavgm import FedAvgM
from src.algorithms.fedopt import FedOpt
from src.algorithms.fed_nolowe import FedNoLowe


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def print_data_distribution(client_loaders, num_classes=10):
    """Print the data distribution across clients."""
    print("\n" + "=" * 70)
    print("Data Distribution Across Clients")
    print("=" * 70)
    
    for i, loader in enumerate(client_loaders):
        class_counts = [0] * num_classes
        for _, labels in loader:
            for label in labels:
                class_counts[label.item()] += 1
        
        total = sum(class_counts)
        non_zero_classes = [c for c in range(num_classes) if class_counts[c] > 0]
        print(f"Client {i+1}: {total} samples, Classes: {non_zero_classes}")
    print("=" * 70 + "\n")


def train_method(method_name, model, client_loaders, test_loader, device, config):
    """Train a single federated learning method."""
    print(f"\n{'=' * 70}")
    print(f"Training with {method_name}")
    print(f"{'=' * 70}")
    
    if method_name == 'FedAvg':
        algorithm = FedAvg(
            model, client_loaders, test_loader, device,
            local_epochs=config['local_epochs'],
            learning_rate=config['learning_rate']
        )
    elif method_name == 'FedAvgM':
        algorithm = FedAvgM(
            model, client_loaders, test_loader, device,
            local_epochs=config['local_epochs'],
            learning_rate=config['learning_rate'],
            server_momentum=config.get('server_momentum', 0.9)
        )
    elif method_name == 'FedOpt':
        algorithm = FedOpt(
            model, client_loaders, test_loader, device,
            local_epochs=config['local_epochs'],
            learning_rate=config['learning_rate'],
            server_lr=config.get('server_lr', 1.0),
            server_optimizer=config.get('server_optimizer', 'adam'),
            beta1=config.get('beta1', 0.9),
            beta2=config.get('beta2', 0.99),
            tau=config.get('tau', 1e-3)
        )
    elif method_name == 'FedNoLowe':
        algorithm = FedNoLowe(
            model, client_loaders, test_loader, device,
            local_epochs=config['local_epochs'],
            learning_rate=config['learning_rate']
        )
    else:
        raise ValueError(f"Unknown method: {method_name}")
    
    history = algorithm.train(
        num_rounds=config['num_rounds'],
        client_fraction=config['client_fraction']
    )
    
    return history


def plot_results(results, dataset_name, distribution, save_path=None):
    """Plot training results comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy
    ax1 = axes[0]
    for method_name, history in results.items():
        ax1.plot(history['accuracy'], label=method_name, linewidth=2)
    ax1.set_xlabel('Communication Round', fontsize=12)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax1.set_title(f'{dataset_name.upper()} - {distribution.upper()} - Accuracy', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2 = axes[1]
    for method_name, history in results.items():
        ax2.plot(history['loss'], label=method_name, linewidth=2)
    ax2.set_xlabel('Communication Round', fontsize=12)
    ax2.set_ylabel('Test Loss', fontsize=12)
    ax2.set_title(f'{dataset_name.upper()} - {distribution.upper()} - Loss', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = f'{dataset_name}_{distribution}_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.close()


def print_final_results(results, dataset_name, distribution):
    """Print final results summary."""
    print("\n" + "=" * 70)
    print(f"FINAL RESULTS - {dataset_name.upper()} - {distribution.upper()}")
    print("=" * 70)
    print(f"{'Method':<15} {'Final Acc (%)':<15} {'Best Acc (%)':<15} {'Final Loss':<15}")
    print("-" * 70)
    
    for method_name, history in results.items():
        final_acc = history['accuracy'][-1]
        best_acc = max(history['accuracy'])
        final_loss = history['loss'][-1]
        print(f"{method_name:<15} {final_acc:<15.2f} {best_acc:<15.2f} {final_loss:<15.4f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Federated Learning Training')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'cifar10'],
                        help='Dataset to use (default: mnist)')
    parser.add_argument('--distribution', type=str, default='iid',
                        choices=['iid', 'non-iid'],
                        help='Data distribution (default: iid)')
    parser.add_argument('--method', type=str, default='all',
                        choices=['all', 'fedavg', 'fedavgm', 'fedopt', 'fednolowe'],
                        help='Method to train (default: all)')
    parser.add_argument('--num-clients', type=int, default=10,
                        help='Number of clients (default: 10)')
    parser.add_argument('--num-rounds', type=int, default=50,
                        help='Number of communication rounds (default: 50)')
    parser.add_argument('--local-epochs', type=int, default=5,
                        help='Number of local epochs (default: 5)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--client-fraction', type=float, default=1.0,
                        help='Fraction of clients per round (default: 1.0)')
    parser.add_argument('--server-momentum', type=float, default=0.9,
                        help='Server momentum for FedAvgM (default: 0.9)')
    parser.add_argument('--server-lr', type=float, default=0.01,
                        help='Server learning rate for FedOpt (default: 0.01)')
    parser.add_argument('--server-optimizer', type=str, default='adam',
                        choices=['sgd', 'sgdm', 'adam', 'adagrad', 'yogi'],
                        help='Server optimizer for FedOpt (default: adam)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Disable plotting')

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'=' * 70}")
    print(f"FEDERATED LEARNING TRAINING")
    print(f"{'=' * 70}")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Distribution: {args.distribution.upper()}")
    print(f"Device: {device}")
    print(f"Number of clients: {args.num_clients}")
    print(f"Communication rounds: {args.num_rounds}")
    print(f"Local epochs: {args.local_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"{'=' * 70}")

    # Load dataset
    print(f"\nLoading {args.dataset.upper()} dataset...")
    train_dataset, test_dataset = load_dataset(args.dataset)

    # Create federated data
    iid = (args.distribution == 'iid')
    print(f"Creating federated data ({'IID' if iid else 'Non-IID'} distribution)...")
    client_datasets = create_federated_data(train_dataset, args.num_clients, iid=iid)
    client_loaders, test_loader = get_data_loaders(client_datasets, test_dataset, args.batch_size)

    print(f"Total training samples: {len(train_dataset)}")
    print(f"Total test samples: {len(test_dataset)}")
    print(f"Samples per client: ~{len(train_dataset) // args.num_clients}")

    # Print data distribution
    print_data_distribution(client_loaders)

    # Training configuration
    config = {
        'num_rounds': args.num_rounds,
        'local_epochs': args.local_epochs,
        'learning_rate': args.learning_rate,
        'client_fraction': args.client_fraction,
        'server_momentum': args.server_momentum,
        'server_lr': args.server_lr,
        'server_optimizer': args.server_optimizer,
        'beta1': 0.9,
        'beta2': 0.99,
        'tau': 1e-3
    }

    # Determine which methods to train
    if args.method == 'all':
        methods = ['FedAvg', 'FedAvgM', 'FedOpt', 'FedNoLowe']
    else:
        methods = [args.method.replace('fedavg', 'FedAvg').replace('fedavgm', 'FedAvgM').replace('fedopt', 'FedOpt').replace('fednolowe', 'FedNoLowe')]

    # Train each method
    results = {}
    for method_name in methods:
        model = UnifiedCNN(dataset=args.dataset)
        history = train_method(method_name, model, client_loaders, test_loader, device, config)
        results[method_name] = history

    # Print final results
    print_final_results(results, args.dataset, args.distribution)

    # Plot results
    if not args.no_plot and len(results) > 0:
        plot_results(results, args.dataset, args.distribution)


if __name__ == '__main__':
    main()


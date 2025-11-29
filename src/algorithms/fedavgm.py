"""
Federated Averaging with Momentum (FedAvgM) algorithm.

FedAvgM extends FedAvg by adding server-side momentum to accelerate convergence.
Reference: Hsu et al., "Measuring the Effects of Non-Identical Data Distribution for
Federated Visual Classification" (2019)
"""
import torch
import copy
from src.utils.training_utils import train_local_model, evaluate_model


class FedAvgM:
    """Federated Averaging with Momentum algorithm."""
    
    def __init__(self, model, client_loaders, test_loader, device,
                 local_epochs=5, learning_rate=0.01, server_momentum=0.9):
        """
        Initialize FedAvgM algorithm.
        
        Args:
            model: The global model
            client_loaders: List of data loaders for clients
            test_loader: DataLoader for test data
            device: Device to train on
            local_epochs: Number of local training epochs
            learning_rate: Learning rate for local training
            server_momentum: Momentum coefficient for server update (default: 0.9)
        """
        self.global_model = model.to(device)
        self.client_loaders = client_loaders
        self.test_loader = test_loader
        self.device = device
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.server_momentum = server_momentum
        self.num_clients = len(client_loaders)
        
        # Initialize server momentum buffer
        self.momentum_buffer = None
        
    def _init_momentum_buffer(self):
        """Initialize momentum buffer with zeros."""
        self.momentum_buffer = {}
        for key, param in self.global_model.state_dict().items():
            self.momentum_buffer[key] = torch.zeros_like(param)
    
    def train_round(self, client_fraction=1.0):
        """
        Execute one round of federated learning with momentum.
        
        Args:
            client_fraction: Fraction of clients to select for training
            
        Returns:
            accuracy: Test accuracy after this round
            loss: Test loss after this round
        """
        # Initialize momentum buffer on first round
        if self.momentum_buffer is None:
            self._init_momentum_buffer()
        
        # Select clients
        num_selected = max(1, int(self.num_clients * client_fraction))
        selected_clients = torch.randperm(self.num_clients)[:num_selected].tolist()
        
        # Store old model parameters
        old_params = {key: param.clone() for key, param in self.global_model.state_dict().items()}
        
        # Train local models
        client_models = []
        client_weights = []
        
        for client_id in selected_clients:
            # Create local model (copy of global model)
            local_model = copy.deepcopy(self.global_model)
            
            # Train on client data
            local_model, _ = train_local_model(
                local_model,
                self.client_loaders[client_id],
                self.local_epochs,
                self.learning_rate,
                self.device
            )
            
            client_models.append(local_model)
            client_weights.append(len(self.client_loaders[client_id].dataset))
        
        # Normalize weights
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]
        
        # Compute weighted average of client models (pseudo-gradient)
        avg_params = {}
        for key in old_params.keys():
            avg_params[key] = torch.zeros_like(old_params[key])
            for client_model, weight in zip(client_models, client_weights):
                avg_params[key] += weight * client_model.state_dict()[key]
        
        # Compute update (delta) = avg_params - old_params
        # Update momentum: v = momentum * v + delta
        # New params = old_params + v
        global_dict = self.global_model.state_dict()
        for key in global_dict.keys():
            delta = avg_params[key] - old_params[key]
            self.momentum_buffer[key] = self.server_momentum * self.momentum_buffer[key] + delta
            global_dict[key] = old_params[key] + self.momentum_buffer[key]
        
        self.global_model.load_state_dict(global_dict)
        
        # Evaluate
        accuracy, loss = evaluate_model(self.global_model, self.test_loader, self.device)
        
        return accuracy, loss
    
    def train(self, num_rounds, client_fraction=1.0):
        """
        Train the federated model.
        
        Args:
            num_rounds: Number of communication rounds
            client_fraction: Fraction of clients to select per round
            
        Returns:
            history: Dictionary with training history
        """
        history = {
            'accuracy': [],
            'loss': []
        }
        
        for round_num in range(num_rounds):
            accuracy, loss = self.train_round(client_fraction)
            history['accuracy'].append(accuracy)
            history['loss'].append(loss)
            
            print(f"Round {round_num + 1}/{num_rounds} - "
                  f"Test Accuracy: {accuracy:.2f}%, Test Loss: {loss:.4f}")
        
        return history


"""
Federated Optimization (FedOpt) algorithm with adaptive server optimizers.

FedOpt uses adaptive optimizers (Adam, Adagrad, Yogi) on the server side to update
the global model, treating the aggregated update as a pseudo-gradient.

Reference: Reddi et al., "Adaptive Federated Optimization" (ICLR 2021)
"""
import torch
import copy
from src.utils.training_utils import train_local_model, evaluate_model


class FedOpt:
    """Federated Optimization with adaptive server optimizer."""
    
    def __init__(self, model, client_loaders, test_loader, device,
                 local_epochs=5, learning_rate=0.01,
                 server_lr=1.0, server_optimizer='adam',
                 beta1=0.9, beta2=0.99, tau=1e-3):
        """
        Initialize FedOpt algorithm.
        
        Args:
            model: The global model
            client_loaders: List of data loaders for clients
            test_loader: DataLoader for test data
            device: Device to train on
            local_epochs: Number of local training epochs
            learning_rate: Learning rate for local training
            server_lr: Server-side learning rate
            server_optimizer: Server optimizer type ('sgd', 'adam', 'adagrad', 'yogi')
            beta1: First moment coefficient for Adam/Yogi
            beta2: Second moment coefficient for Adam/Yogi
            tau: Adaptivity parameter (epsilon for Adam)
        """
        self.global_model = model.to(device)
        self.client_loaders = client_loaders
        self.test_loader = test_loader
        self.device = device
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.server_lr = server_lr
        self.server_optimizer = server_optimizer.lower()
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau
        self.num_clients = len(client_loaders)
        
        # Initialize optimizer state
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0     # Time step
        
    def _init_optimizer_state(self):
        """Initialize optimizer state (momentum and variance estimates)."""
        self.m = {}
        self.v = {}
        for key, param in self.global_model.state_dict().items():
            self.m[key] = torch.zeros_like(param, dtype=torch.float32)
            self.v[key] = torch.zeros_like(param, dtype=torch.float32)
    
    def _server_update(self, old_params, delta):
        """
        Apply server optimizer update.
        
        Args:
            old_params: Previous model parameters
            delta: Pseudo-gradient (aggregated update)
            
        Returns:
            new_params: Updated parameters
        """
        new_params = {}
        
        if self.server_optimizer == 'sgd':
            # Simple SGD: w = w + server_lr * delta
            for key in old_params.keys():
                new_params[key] = old_params[key] + self.server_lr * delta[key]
                
        elif self.server_optimizer == 'sgdm':
            # SGD with momentum
            for key in old_params.keys():
                self.m[key] = self.beta1 * self.m[key] + delta[key]
                new_params[key] = old_params[key] + self.server_lr * self.m[key]
                
        elif self.server_optimizer == 'adam':
            # FedAdam
            for key in old_params.keys():
                delta_float = delta[key].float()
                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * delta_float
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (delta_float ** 2)
                # Bias correction
                m_hat = self.m[key] / (1 - self.beta1 ** self.t)
                v_hat = self.v[key] / (1 - self.beta2 ** self.t)
                update = self.server_lr * m_hat / (torch.sqrt(v_hat) + self.tau)
                new_params[key] = old_params[key] + update.to(old_params[key].dtype)
                
        elif self.server_optimizer == 'adagrad':
            # FedAdagrad
            for key in old_params.keys():
                delta_float = delta[key].float()
                self.v[key] = self.v[key] + delta_float ** 2
                update = self.server_lr * delta_float / (torch.sqrt(self.v[key]) + self.tau)
                new_params[key] = old_params[key] + update.to(old_params[key].dtype)
                
        elif self.server_optimizer == 'yogi':
            # FedYogi
            for key in old_params.keys():
                delta_float = delta[key].float()
                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * delta_float
                # Yogi update for second moment
                delta_sq = delta_float ** 2
                sign = torch.sign(delta_sq - self.v[key])
                self.v[key] = self.v[key] + (1 - self.beta2) * sign * delta_sq
                update = self.server_lr * self.m[key] / (torch.sqrt(self.v[key]) + self.tau)
                new_params[key] = old_params[key] + update.to(old_params[key].dtype)
        else:
            raise ValueError(f"Unknown server optimizer: {self.server_optimizer}")
            
        return new_params
    
    def train_round(self, client_fraction=1.0):
        """
        Execute one round of federated learning with adaptive server optimizer.
        
        Args:
            client_fraction: Fraction of clients to select for training
            
        Returns:
            accuracy: Test accuracy after this round
            loss: Test loss after this round
        """
        # Initialize optimizer state on first round
        if self.m is None:
            self._init_optimizer_state()
        
        self.t += 1  # Increment time step
        
        # Select clients
        num_selected = max(1, int(self.num_clients * client_fraction))
        selected_clients = torch.randperm(self.num_clients)[:num_selected].tolist()
        
        # Store old model parameters
        old_params = {key: param.clone() for key, param in self.global_model.state_dict().items()}
        
        # Train local models
        client_models = []
        client_weights = []
        
        for client_id in selected_clients:
            local_model = copy.deepcopy(self.global_model)
            local_model, _ = train_local_model(
                local_model, self.client_loaders[client_id],
                self.local_epochs, self.learning_rate, self.device
            )
            client_models.append(local_model)
            client_weights.append(len(self.client_loaders[client_id].dataset))

        # Normalize weights
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]

        # Compute weighted average (pseudo-gradient / delta)
        delta = {}
        for key in old_params.keys():
            avg_param = torch.zeros_like(old_params[key])
            for client_model, weight in zip(client_models, client_weights):
                avg_param += weight * client_model.state_dict()[key]
            delta[key] = avg_param - old_params[key]

        # Apply server optimizer update
        new_params = self._server_update(old_params, delta)
        self.global_model.load_state_dict(new_params)

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


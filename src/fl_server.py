import flwr as fl
from typing import List, Tuple

def get_evaluate_fn():
    """
    In a real implementation, the server uses a held-out global validation set.
    """
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: dict):
        # We don't perform global evaluation in this boilerplate 
        # as it requires a centralized testing dataset which defeats
        # the purpose in some strict privacy setups.
        return None, {}
    return evaluate

def start_fl_server():
    print("Starting Federated Learning Aggregation Server...")
    
    # Define strategy (Federated Averaging)
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
        min_fit_clients=2, # Require at least 2 hospitals to train
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn()
    )

    # Start the server on port 8080
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

if __name__ == "__main__":
    # Note: Requires starting clients in separate processes
    print("Federated Server defined. Waiting for nodes...")

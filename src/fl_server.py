import flwr as fl
from flwr.common import parameters_to_ndarrays
from typing import List, Tuple
import socket
import threading
import time
import os
import numpy as np
import http.server
import socketserver
import socket
import threading
import time

def udp_broadcaster(port=55555, fl_port=8080, http_port=8081):
    """Continuously broadcasts the presence of the FL Server on the local network."""
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    
    message = f"DACNET_FL_SERVER:{fl_port}:{http_port}".encode('utf-8')
    while True:
        try:
            server.sendto(message, ('<broadcast>', port))
        except Exception:
            try:
                server.sendto(message, ('255.255.255.255', port))
            except:
                pass
        time.sleep(2)

def http_file_server(port=8081):
    """Hosts the data directory so clients can download global_model.npz."""
    os.chdir(os.path.join(os.path.dirname(__file__), '..', 'data'))
    handler = http.server.SimpleHTTPRequestHandler
    class LoggingHandler(handler):
        def log_message(self, format, *args):
            if "global_model.npz" in args[0]:
                print(f"--> [TRANSACTION LOG] A Hospital Node ({self.client_address[0]}) just downloaded the latest Global AI Model!")
    
    with socketserver.TCPServer(("", port), LoggingHandler) as httpd:
        print(f"HTTP Global Model File Server running on port {port}")
        httpd.serve_forever()

def start_background_services():
    t1 = threading.Thread(target=udp_broadcaster, daemon=True)
    t1.start()
    t2 = threading.Thread(target=http_file_server, daemon=True)
    t2.start()

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

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round: int, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            print(f"Saving global aggregated model for round {server_round}...")
            # Convert aggregated parameters back to numpy arrays
            ndarrays = parameters_to_ndarrays(aggregated_parameters)
            
            # Save the ndarrays to a .npz file
            save_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'global_model.npz')
            np.savez(save_path, *ndarrays)
            print(f"Global model saved to {save_path}. Ready for hospital download!")
            
        return aggregated_parameters, aggregated_metrics

def start_fl_server():
    print("Starting Federated Learning Aggregation Server...")
    print("Initiating Auto-Discovery UDP Beacon & File Server...")
    start_background_services()
    
    # Define strategy (Custom Federated Averaging that saves the model)
    strategy = SaveModelStrategy(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
        min_fit_clients=1, # Require at least 1 hospital to train
        min_evaluate_clients=1,
        min_available_clients=1,
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
    start_fl_server()

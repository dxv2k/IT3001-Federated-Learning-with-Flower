from typing import Optional

import flwr as fl
import torch
from flwr.common import Metrics
from flwr.server import ServerConfig
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg
from torchvision.models import mobilenet_v2

from src.utils.dataset_utils import load_datasets
from src.utils.helper_func import set_parameters, test


# The `evaluate` function will be by Flower called after every round
def evaluate(
    server_round: int,
    parameters: fl.common.NDArrays,
    config: dict[str, fl.common.Scalar],
) -> Optional[tuple[float, dict[str, fl.common.Scalar]]]:
    net = mobilenet_v2(pretrained=False, num_classes=10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device).to(torch.float32)

    _, test_dataloader = load_datasets(batch_size=32)

    set_parameters(net, parameters)  # Update model with the latest parameters
    loss, accuracy, _ = test(net, test_dataloader, device=device)

    print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
    return loss, {"accuracy": accuracy}


def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Perform two rounds of training with one local epoch, increase to two local
    epochs afterwards.
    """
    config = {
        "server_round": server_round,  # The current round of federated learning
        "local_epochs": 1,  # if server_round < 2 else 2,  #
    }
    return config


# TODO: saved model after aggregate
def main():
    client_manager = SimpleClientManager()
    # client_manager = (SimpleClientManager([client]),)
    print("Number of clients: ", len(client_manager))

    strategy = FedAvg(
        # fraction_fit=1,
        # fraction_evaluate=1,
        # min_fit_clients=2,
        # min_evaluate_clients=2,
        # min_available_clients=NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=weighted_average,  # Pass the metric aggregation function
        # initial_parameters=fl.common.ndarrays_to_parameters(params),
        evaluate_fn=evaluate,  # Pass the evaluate function to the server
        on_fit_config_fn=fit_config,  # Pass the fit_config function to the server
    )

    # Start the Flower server and connect the client
    server_config = ServerConfig(num_rounds=10)
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=server_config,
        client_manager=client_manager,
        strategy=strategy,
    )
    print("Number of clients: ", len(client_manager))


if __name__ == "__main__":
    main()

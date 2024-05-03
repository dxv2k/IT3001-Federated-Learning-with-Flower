import os
import torch
import argparse
import flwr as fl
from typing import Optional

os.environ.get("RAY_memory_monitor_refresh_ms", "0")  # disable kill workers
os.environ.get("RAY_memory_usage_threshold", "0.8")

from flwr.common import Metrics
from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig
from flwr.simulation import start_simulation
from client_device import CifarClient, mobilenet_v2
from flwr.server.client_manager import SimpleClientManager

from src.utils.helper_func import save_metrics_to_csv, set_parameters, test
from src.utils.dataset_utils import load_datasets, load_seperate_datasets


def client_fn(cid: str) -> CifarClient:
    # Load model and data (MobileNetV2, CIFAR-10)
    net = mobilenet_v2(weights=None, num_classes=10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device).to(torch.float32)

    # Load and preprocess your dataset
    train_dataloader, test_dataloader = load_datasets(batch_size=BATCH_SIZE)

    # Create the Flower client
    client = CifarClient(
        client_id=cid,
        net=net,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
    )

    return client


def client_fn_gpu(cid: str):
    global client_train_datasets
    global client_test_datasets

    # Load model and data (MobileNetV2, CIFAR-10)
    net = mobilenet_v2(weights=None, num_classes=10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device).to(torch.float32)

    # Load train and test datasets for the specific client
    train_dataloader = client_train_datasets[int(cid)]
    test_dataloader = client_test_datasets[int(cid)]

    # Create the Flower client
    client = CifarClient(
        client_id=cid,
        net=net,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
    )

    return client


def evaluate(
    server_round: int,
    parameters: fl.common.NDArrays,
    config: dict[str, fl.common.Scalar],
) -> Optional[tuple[float, dict[str, fl.common.Scalar]]]:
    net = mobilenet_v2(pretrained=False, num_classes=10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device).to(torch.float32)

    _, test_dataloader = load_datasets(batch_size=BATCH_SIZE)

    set_parameters(net, parameters)  # Update model with the latest parameters
    loss, accuracy, metrics_list = test(
        net, test_dataloader, round=server_round, device=device
    )
    print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
    save_metrics_to_csv("server_eval_metrics.csv", metrics_list)

    return loss, {"accuracy": accuracy}


def metrics_weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def fit_config(server_round: int, local_epochs: int = 1):
    """Return training configuration dict for each round.

    Perform two rounds of training with one local epoch, increase to two local
    epochs afterwards.
    """
    config = {
        "server_round": server_round,  # The current round of federated learning
        "local_epochs": local_epochs,
    }
    return config


def main(
    num_cpus: int = 4,
    num_gpus: int = 1,
    max_cpu_utilized: float = 0.05,
    max_gpu_utilized: float = 0.1,
    num_rounds: int = 50,
    num_clients: int = 2,
):
    strategy = FedAvg(
        # fraction_fit=1,
        # fraction_evaluate=1,
        # min_fit_clients=2,
        # min_evaluate_clients=2,
        # min_available_clients=NUM_CLIENTS,
        # Pass the metric aggregation function
        evaluate_metrics_aggregation_fn=metrics_weighted_average,
        # initial_parameters=fl.common.ndarrays_to_parameters(params),
        evaluate_fn=evaluate,  # Pass the evaluate function to the server
        on_fit_config_fn=fit_config,  # Pass the fit_config function to the server
    )

    server_config = ServerConfig(num_rounds=num_rounds)

    client_resources = {"num_cpus": max_cpu_utilized, "num_gpus": max_gpu_utilized}

    # Specify number of FL rounds
    client_manager = SimpleClientManager()

    # Launch the simulation
    history = start_simulation(
        client_fn=client_fn_gpu,  # A function to run a _virtual_ client when required
        num_clients=num_clients,  # Total number of clients available
        config=server_config,
        strategy=strategy,  # A Flower strategy
        client_resources=client_resources,
        client_manager=client_manager,
        ray_init_args={
            "include_dashboard": True,  # we need this one for tracking
            "num_cpus": num_cpus,
            "num_gpus": num_gpus,
        },
    )

    return history


def parse_args():
    parser = argparse.ArgumentParser(description="Federated Learning Simulation")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_clients", type=int, default=2, help="Number of clients")
    parser.add_argument("--num_rounds", type=int, default=50, help="Number of rounds")
    parser.add_argument(
        "--local_epochs", type=int, default=1, help="Number of local epochs"
    )
    parser.add_argument("--num_cpus", type=int, default=4, help="Number of CPUs")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument(
        "--max_cpu_utilized", type=float, default=0.05, help="Max CPU utilization"
    )
    parser.add_argument(
        "--max_gpu_utilized", type=float, default=0.1, help="Max GPU utilization"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    BATCH_SIZE = args.batch_size
    NUM_CLIENTS = args.num_clients
    NUM_ROUNDS = args.num_rounds
    LOCAL_EPOCHS = args.local_epochs

    # NOTE: my client resources
    # client get 5% of the CPU & 10% GPU because
    # estimate from Raspberrypi 4GB to RTX 2070 & Ryzen 5 2600
    client_train_datasets, client_test_datasets = load_seperate_datasets(
        NUM_CLIENTS, BATCH_SIZE
    )

    result = main(
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        max_cpu_utilized=args.max_cpu_utilized,
        max_gpu_utilized=args.max_gpu_utilized,
        num_rounds=NUM_ROUNDS,
        num_clients=NUM_CLIENTS,
    )

    # TODO: save result to csv

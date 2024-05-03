import os
import random

import flwr as fl
import torch
from torchvision.models import mobilenet_v2

from src.utils.dataset_utils import load_datasets
from src.utils.helper_func import (
    get_parameters,
    save_metrics_to_csv,
    set_parameters,
    test,
    train,
)

# Make PyTorch log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Define the Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self, client_id, net, train_dataloader, test_dataloader):
        self.client_id = client_id
        self.net = net
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.curr_round = 0

    def get_parameters(self, config):
        # server_round = config["server_round"]
        # print(
        #     f"[Client, round {server_round}] get_parameters, config: {config}"
        # )

        return get_parameters(self.net)

    def set_parameters(self, parameters, config):
        # server_round = config["server_round"]
        # print(
        #     f"[Client, round {server_round}] set_parameters, config: {config}"
        # )

        set_parameters(self.net, parameters)

    def fit(self, parameters, config):
        self.curr_round = int(config["server_round"])
        local_epochs = int(config["local_epochs"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Client, round {self.curr_round}] fit, config: {config}")

        self.set_parameters(parameters, config)
        metrics_list = train(
            self.net,
            self.train_dataloader,
            epochs=local_epochs,
            round=self.curr_round,
            device=device,
        )
        save_metrics_to_csv(
            f"client_{self.client_id}_train_metrics.csv", metrics_list
        )
        torch.save(
            self.net.state_dict(),
            f"./models/client_{self.client_id}_round_{self.curr_round}_model_weights.pth",
        )

        return self.get_parameters({}), len(self.train_dataloader.dataset), {}

    def evaluate(self, parameters, config):
        # server_round = config["server_round"] # BUG:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Client, round {self.curr_round}] evaluate, config: {config}")

        self.set_parameters(parameters, config)
        loss, accuracy, metrics_list = test(
            self.net, self.test_dataloader, round=self.curr_round, device=device
        )
        print(f"Client-side evaluation loss {loss} / accuracy {accuracy}")
        save_metrics_to_csv(
            f"client_{self.client_id}_eval_metrics.csv", metrics_list
        )

        return (
            float(loss),
            len(self.test_dataloader.dataset),
            {"accuracy": float(accuracy)},
        )


def main():
    # Load model and data (MobileNetV2, CIFAR-10)
    net = mobilenet_v2(weights=None, num_classes=10)
    net.to(torch.float32)

    # Load and preprocess your dataset
    train_dataloader, test_dataloader = load_datasets(batch_size=32)

    # Create the Flower client
    client = CifarClient(
        client_id=random.randint(0, 1000),
        net=net,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
    )

    # Start the training process
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)

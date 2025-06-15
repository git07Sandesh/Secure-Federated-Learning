"""my-first-federated-learning: A Flower / PyTorch app."""

import torch

import logging
logging.basicConfig(level=logging.INFO, format="[Client %(name)s] %(message)s")
# We need this to print logs from the client. Why can't we use simple print? -> flwr run . creates multiple subprocesses that creates an isolated
# environment for each client, so print statements won't show up in the main process. So we need to use logging to ensure that logs from each client are captured and displayed correctly.

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from my_first_federated_learning.task import Net, get_weights, load_data, set_weights, test, train


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(self, parameters, config):
        logging.info(f"Client Starting fitting")   
        set_weights(self.net, parameters)

        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
        )
        logging.info(f"Training loss: {train_loss}")
        logging.info(f"Client Finished fitting")
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        logging.info(f"Evaluating model...")
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        logging.info(f"Evaluation loss: {loss}, accuracy: {accuracy}")
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    # Load model and data
    net = Net()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)

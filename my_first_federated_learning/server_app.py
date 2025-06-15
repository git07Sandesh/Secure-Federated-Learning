"""my-first-federated-learning: A Flower / PyTorch app."""
import matplotlib as plt
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, Krum, FedMedian
from flwr.common import Parameters
from my_first_federated_learning.task import Net, get_weights
import wandb
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[Server] %(message)s'
)


class StrategyWithLossLogger(FedAvg):
    def __init__(self, log_file="loss_log.txt", **kwargs):
        super().__init__(**kwargs)
        self.log_file = log_file

    def aggregate_evaluate(self, rnd, results, failures):
        aggregated_loss, metrics = super().aggregate_evaluate(rnd, results, failures)
        if aggregated_loss is not None:
            logging.info(f"[Server] Round {rnd} aggregated loss: {aggregated_loss:.4f}")
            with open(self.log_file, "a") as f:
                f.write(f"{rnd},{aggregated_loss:.4f}\n")
        return aggregated_loss, metrics
    
def server_fn(context: Context):
    strategy_name = context.run_config.get("strategy", "FedAvg")
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    initial_parameters = ndarrays_to_parameters(get_weights(Net()))


    wandb.init(project="flower-debug", name="manual-test")
    wandb.log({"round": 0, "dummy_accuracy": 0.1})
    # strategy_map = {
    #     "FedAvg": FedAvg,
    #     "Krum": Krum,
    #     "Median": FedMedian,
    # }

    # strategy_cls = strategy_map.get(strategy_name, FedAvg)

    # strategy = StrategyWithLossLogger(
    #     log_file=f"loss_log_{strategy_name}.txt",
    #     fraction_fit=fraction_fit,
    #     fraction_evaluate=1.0,
    #     min_available_clients=2,
    #     initial_parameters=initial_parameters,
    # )

    strategy = Krum(
        fraction_fit=fraction_fit,
        fraction_evaluate=0.8,
        min_available_clients=2,
        num_malicious_clients=3,
        initial_parameters=initial_parameters,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)

import os
from torch.utils.tensorboard import SummaryWriter


class TrainingLogger:
    def __init__(self, log_dir: str, run_name: str):
        self.run_dir = os.path.join(log_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.writer = SummaryWriter(self.run_dir)
        self.log_file = open(os.path.join(self.run_dir, "train_log.txt"), "a")

    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    def log_metrics(self, metrics: dict, step: int, prefix: str = ""):
        for k, v in metrics.items():
            self.writer.add_scalar(f"{prefix}{k}", v, step)

    def log_text(self, message: str):
        print(message)
        self.log_file.write(message + "\n")
        self.log_file.flush()

    def close(self):
        self.writer.close()
        self.log_file.close()

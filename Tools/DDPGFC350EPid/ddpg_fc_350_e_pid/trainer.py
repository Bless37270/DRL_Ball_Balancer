import argparse
import sys
from pathlib import Path
from typing import Optional

from .optimizer_torch import DDPGFC350EPidSettings


class DDPGFC350EPidTrainer:
    trainer_type = "ddpg_fc_350_e_pid"

    def __init__(
        self,
        env_path: Optional[str],
        config_path: str = "Tools/DDPGFC350EPid/config.yaml",
        output_dir: str = "results",
        num_envs: int = 1,
        force_cpu: bool = False,
    ):
        self.env_path = env_path
        self.config_path = config_path
        self.output_dir = output_dir
        self.num_envs = num_envs
        self.force_cpu = force_cpu
        self.settings = DDPGFC350EPidSettings()

    def train(self) -> None:
        package_root = Path(__file__).resolve().parents[1]
        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))

        from ddpg_fc_350_e_pid_train import train

        args = argparse.Namespace(
            env=self.env_path,
            config=self.config_path,
            output_dir=self.output_dir,
            num_envs=self.num_envs,
            cpu=self.force_cpu,
        )
        train(args)


def main() -> None:
    parser = argparse.ArgumentParser(description="DDPG-FC-350-E-PID trainer entry point.")
    parser.add_argument("--env", default=None)
    parser.add_argument("--config", default="Tools/DDPGFC350EPid/config.yaml")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    trainer = DDPGFC350EPidTrainer(
        env_path=args.env,
        config_path=str(Path(args.config)),
        output_dir=args.output_dir,
        num_envs=args.num_envs,
        force_cpu=args.cpu,
    )
    trainer.train()


if __name__ == "__main__":
    main()

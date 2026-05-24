import argparse
import json
import random
import shutil
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.stats_side_channel import StatsAggregationMethod, StatsSideChannel
from torch.utils.tensorboard import SummaryWriter


@dataclass
class DDPGConfig:
    behavior_name: str
    run_id: str
    seed: int
    hidden_units: int
    num_layers: int
    max_steps: int
    start_steps: int
    update_after: int
    update_every: int
    batch_size: int
    replay_buffer_size: int
    gamma: float
    tau: float
    actor_learning_rate: float
    critic_learning_rate: float
    exploration_noise: float
    exploration_noise_min: float
    exploration_noise_decay_steps: int
    action_l2_weight: float
    summary_freq: int
    save_freq: int
    progress_step_mode: str
    worker_id: int
    num_envs: int
    no_graphics: bool
    timeout_wait: int
    width: int
    height: int
    quality_level: int
    time_scale: float
    target_frame_rate: int
    capture_frame_rate: int


class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_units: int, num_layers: int):
        super().__init__()
        layers = []
        last_size = input_size
        for _ in range(num_layers):
            layers.append(nn.Linear(last_size, hidden_units))
            layers.append(nn.ELU())
            last_size = hidden_units
        layers.append(nn.Linear(last_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Actor(nn.Module):
    def __init__(self, obs_size: int, action_size: int, hidden_units: int, num_layers: int):
        super().__init__()
        self.body = MLP(obs_size, action_size, hidden_units, num_layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.body(obs))


class MLAgentsActorExport(nn.Module):
    def __init__(self, actor: Actor, action_size: int):
        super().__init__()
        self.actor = actor
        self.register_buffer("version_number", torch.tensor([3.0], dtype=torch.float32))
        self.register_buffer("memory_size", torch.tensor([0.0], dtype=torch.float32))
        self.register_buffer("continuous_action_output_shape", torch.tensor([float(action_size)], dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        actions = self.actor(obs)
        return (
            self.version_number,
            self.memory_size,
            actions,
            self.continuous_action_output_shape,
            actions,
        )


class Critic(nn.Module):
    def __init__(self, obs_size: int, action_size: int, hidden_units: int, num_layers: int):
        super().__init__()
        self.q = MLP(obs_size + action_size, 1, hidden_units, num_layers)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q(torch.cat([obs, action], dim=-1))


class ReplayBuffer:
    def __init__(self, capacity: int, obs_size: int, action_size: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_size), dtype=np.float32)
        self.actions = np.zeros((capacity, action_size), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_size), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.index = 0
        self.size = 0

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool) -> None:
        self.obs[self.index] = obs
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_obs[self.index] = next_obs
        self.dones[self.index] = float(done)
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx], device=device),
            torch.as_tensor(self.actions[idx], device=device),
            torch.as_tensor(self.rewards[idx], device=device),
            torch.as_tensor(self.next_obs[idx], device=device),
            torch.as_tensor(self.dones[idx], device=device),
        )


def load_config(path: Path) -> DDPGConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    top_level_run_id = raw.get("run_id")
    if "behaviors" in raw:
        behavior_name, behavior = next(iter(raw["behaviors"].items()))
        hyperparameters = behavior.get("hyperparameters", {})
        network = behavior.get("network_settings", {})
        training = {
            **hyperparameters,
            "max_steps": behavior.get("max_steps", 3500000),
            "summary_freq": behavior.get("summary_freq", 10000),
            "save_freq": behavior.get("save_freq", 100000),
            "progress_step_mode": behavior.get("progress_step_mode", raw.get("progress_step_mode", "environment")),
        }
        environment = raw.get("environment", {})
        raw = {
            "behavior_name": behavior_name,
            "run_id": behavior.get("run_id", top_level_run_id or behavior_name),
            "seed": raw.get("seed", 1),
            "network": network,
            "training": training,
            "environment": environment,
        }
    else:
        network = raw["network"]
        training = raw["training"]
        environment = raw["environment"]

    if network.get("activation", "elu").lower() != "elu":
        raise ValueError("DDPG-FC-350-E-PID requires ELU activation.")

    return DDPGConfig(
        behavior_name=raw["behavior_name"],
        run_id=raw.get("run_id", "ddpg_fc_350_e_pid"),
        seed=int(raw.get("seed", 1)),
        hidden_units=int(network.get("hidden_units", 350)),
        num_layers=int(network.get("num_layers", 2)),
        max_steps=int(training.get("max_steps", 3500000)),
        start_steps=int(training.get("start_steps", 5000)),
        update_after=int(training.get("update_after", 5000)),
        update_every=int(training.get("update_every", 1)),
        batch_size=int(training.get("batch_size", 256)),
        replay_buffer_size=int(training.get("replay_buffer_size", training.get("buffer_size", 1000000))),
        gamma=float(training.get("gamma", 0.99)),
        tau=float(training.get("tau", 0.005)),
        actor_learning_rate=float(training.get("actor_learning_rate", 0.0001)),
        critic_learning_rate=float(training.get("critic_learning_rate", 0.001)),
        exploration_noise=float(training.get("exploration_noise", 0.2)),
        exploration_noise_min=float(training.get("exploration_noise_min", 0.02)),
        exploration_noise_decay_steps=int(training.get("exploration_noise_decay_steps", 1500000)),
        action_l2_weight=float(training.get("action_l2_weight", 0.0001)),
        summary_freq=int(training.get("summary_freq", 10000)),
        save_freq=int(training.get("save_freq", 100000)),
        progress_step_mode=str(training.get("progress_step_mode", "environment")).lower(),
        worker_id=int(environment.get("worker_id", 0)),
        num_envs=int(environment.get("num_envs", 1)),
        no_graphics=bool(environment.get("no_graphics", True)),
        timeout_wait=int(environment.get("timeout_wait", 60)),
        width=int(environment.get("width", 84)),
        height=int(environment.get("height", 84)),
        quality_level=int(environment.get("quality_level", 1)),
        time_scale=float(environment.get("time_scale", 20.0)),
        target_frame_rate=int(environment.get("target_frame_rate", -1)),
        capture_frame_rate=int(environment.get("capture_frame_rate", 60)),
    )


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(tau * source_param.data)


def hard_update(source: nn.Module, target: nn.Module) -> None:
    target.load_state_dict(source.state_dict())


def flatten_vector_obs(obs_list: Iterable[np.ndarray]) -> np.ndarray:
    vectors = [obs for obs in obs_list if len(obs.shape) == 2]
    if not vectors:
        raise RuntimeError("DDPG trainer requires at least one vector observation.")
    return np.concatenate(vectors, axis=1).astype(np.float32)


def noise_scale(step: int, cfg: DDPGConfig) -> float:
    progress = min(max(step / max(cfg.exploration_noise_decay_steps, 1), 0.0), 1.0)
    return cfg.exploration_noise + (cfg.exploration_noise_min - cfg.exploration_noise) * progress


def make_engine_channel(cfg: DDPGConfig) -> EngineConfigurationChannel:
    engine_channel = EngineConfigurationChannel()
    engine_channel.set_configuration_parameters(
        width=cfg.width,
        height=cfg.height,
        quality_level=cfg.quality_level,
        time_scale=cfg.time_scale,
        target_frame_rate=cfg.target_frame_rate,
        capture_frame_rate=cfg.capture_frame_rate,
    )
    return engine_channel


def get_progress_step(mode: str, decision_step: int, environment_step: int) -> int:
    if mode == "decision":
        return decision_step
    return environment_step


def collect_environment_stats(
    stats_channels: Iterable[StatsSideChannel],
    pending_stats: Dict[str, list],
) -> None:
    for stats_channel in stats_channels:
        for stat_name, value_list in stats_channel.get_and_reset_stats().items():
            pending_stats[stat_name].extend(value_list)


def write_environment_stats(
    summary_writer: SummaryWriter,
    pending_stats: Dict[str, list],
    step: int,
) -> None:
    for stat_name, value_list in pending_stats.items():
        if not value_list:
            continue

        average_values = [val for val, agg in value_list if agg == StatsAggregationMethod.AVERAGE]
        sum_values = [val for val, agg in value_list if agg == StatsAggregationMethod.SUM]
        most_recent_values = [val for val, agg in value_list if agg == StatsAggregationMethod.MOST_RECENT]
        histogram_values = [val for val, agg in value_list if agg == StatsAggregationMethod.HISTOGRAM]

        if average_values:
            summary_writer.add_scalar(stat_name, float(np.mean(average_values)), step)
        if sum_values:
            summary_writer.add_scalar(stat_name, float(np.sum(sum_values)), step)
        if most_recent_values:
            summary_writer.add_scalar(stat_name, float(most_recent_values[-1]), step)
        if histogram_values:
            summary_writer.add_histogram(stat_name, np.asarray(histogram_values, dtype=np.float32), step)

    pending_stats.clear()


def save_checkpoint(
    output_dir: Path,
    actor: Actor,
    critic: Critic,
    actor_target: Actor,
    critic_target: Critic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    step: int,
    obs_size: int,
    action_size: int,
    cfg: DDPGConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "step": step,
        "obs_size": obs_size,
        "action_size": action_size,
        "config": cfg.__dict__,
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "actor_target": actor_target.state_dict(),
        "critic_target": critic_target.state_dict(),
        "actor_optimizer": actor_optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
    }
    torch.save(checkpoint, output_dir / "checkpoint.pt")
    torch.save(actor.state_dict(), output_dir / f"actor_step_{step}.pt")

    actor.eval()
    export_actor = MLAgentsActorExport(actor, action_size).eval()
    dummy_obs = torch.zeros(1, obs_size, dtype=torch.float32)
    torch.onnx.export(
        export_actor,
        dummy_obs,
        output_dir / f"{cfg.behavior_name}.onnx",
        input_names=["obs_0"],
        output_names=[
            "version_number",
            "memory_size",
            "continuous_actions",
            "continuous_action_output_shape",
            "deterministic_continuous_actions",
        ],
        dynamic_axes={
            "obs_0": {0: "batch"},
            "continuous_actions": {0: "batch"},
            "deterministic_continuous_actions": {0: "batch"},
        },
        opset_version=11,
    )
    actor.train()


def write_configuration_snapshot(output_dir: Path, source_config_path: Path, cfg: DDPGConfig) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "configuration.yaml"
    raw = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    if "behaviors" in raw:
        behavior_name, behavior = next(iter(raw["behaviors"].items()))
        behavior["run_id"] = cfg.run_id
        raw["behaviors"] = {behavior_name: behavior}
    else:
        raw["run_id"] = cfg.run_id
    destination.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def write_training_status(
    output_dir: Path,
    status: Dict[str, object],
) -> None:
    run_logs_dir = output_dir / "run_logs"
    run_logs_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_logs_dir / "training_status.json"
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")


def train(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    cfg = load_config(config_path)
    if args.run_id:
        cfg.run_id = args.run_id
    if args.seed is not None:
        cfg.seed = args.seed
    if args.time_scale is not None:
        cfg.time_scale = args.time_scale
    if cfg.progress_step_mode not in {"environment", "decision"}:
        raise ValueError("progress_step_mode must be 'environment' or 'decision'.")
    if cfg.update_after > cfg.replay_buffer_size:
        raise ValueError(
            f"update_after ({cfg.update_after}) must be less than or equal to "
            f"the replay buffer size ({cfg.replay_buffer_size})."
        )
    if cfg.batch_size > cfg.replay_buffer_size:
        raise ValueError(
            f"batch_size ({cfg.batch_size}) must be less than or equal to "
            f"the replay buffer size ({cfg.replay_buffer_size})."
        )

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    num_envs = max(int(args.num_envs or cfg.num_envs), 1)
    run_output_dir = Path(args.output_dir) / cfg.run_id
    if run_output_dir.exists():
        if args.force:
            shutil.rmtree(run_output_dir)
        else:
            raise FileExistsError(
                f"Results directory '{run_output_dir}' already exists. "
                "Use --force or choose a different --run-id."
            )
    artifact_output_dir = run_output_dir / cfg.behavior_name
    run_logs_dir = run_output_dir / "run_logs"
    write_configuration_snapshot(run_output_dir, config_path, cfg)
    stats_channels = [StatsSideChannel() for _ in range(num_envs)]

    envs = [
        UnityEnvironment(
            file_name=args.env,
            worker_id=cfg.worker_id + env_index,
            seed=cfg.seed + env_index,
            no_graphics=cfg.no_graphics,
            timeout_wait=cfg.timeout_wait,
            log_folder=str(run_logs_dir),
            side_channels=[make_engine_channel(cfg), stats_channels[env_index]],
        )
        for env_index in range(num_envs)
    ]

    episode_returns: Dict[int, float] = {}
    recent_returns: Deque[float] = deque(maxlen=50)
    total_episodes = 0
    last_actor_loss: Optional[float] = None
    last_critic_loss: Optional[float] = None
    best_recent_mean_return = float("-inf")
    training_start_time = time.time()
    summary_writer = SummaryWriter(log_dir=str(artifact_output_dir), flush_secs=30)
    pending_environment_stats: Dict[str, list] = defaultdict(list)

    try:
        for env in envs:
            env.reset()

        behavior_names = []
        for env_index, env in enumerate(envs):
            behavior_name = next((name for name in env.behavior_specs if name.startswith(cfg.behavior_name)), None)
            if behavior_name is None:
                available = ", ".join(env.behavior_specs.keys())
                raise RuntimeError(
                    f"Behavior '{cfg.behavior_name}' not found in env {env_index}. "
                    f"Available behaviors: {available}"
                )
            behavior_names.append(behavior_name)

        behavior_name = behavior_names[0]
        if any(name != behavior_name for name in behavior_names):
            available = ", ".join(behavior_names)
            raise RuntimeError(f"Environments returned different behavior names: {available}")

        spec = envs[0].behavior_specs[behavior_name]
        if not spec.action_spec.is_continuous() or spec.action_spec.continuous_size <= 0:
            raise RuntimeError("DDPG-FC-350-E-PID requires a continuous action space.")

        decision_steps, terminal_steps = envs[0].get_steps(behavior_name)
        obs_size = flatten_vector_obs(decision_steps.obs).shape[1]
        action_size = spec.action_spec.continuous_size

        actor = Actor(obs_size, action_size, cfg.hidden_units, cfg.num_layers).to(device)
        critic = Critic(obs_size, action_size, cfg.hidden_units, cfg.num_layers).to(device)
        actor_target = Actor(obs_size, action_size, cfg.hidden_units, cfg.num_layers).to(device)
        critic_target = Critic(obs_size, action_size, cfg.hidden_units, cfg.num_layers).to(device)
        hard_update(actor, actor_target)
        hard_update(critic, critic_target)

        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.actor_learning_rate)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg.critic_learning_rate)
        replay = ReplayBuffer(cfg.replay_buffer_size, obs_size, action_size)

        last_obs: Dict[Tuple[int, int], np.ndarray] = {}
        last_action: Dict[Tuple[int, int], np.ndarray] = {}
        episode_lengths: Dict[Tuple[int, int], int] = {}
        decision_step = 0
        environment_step = 0

        print(
            f"Training {cfg.behavior_name}: obs={obs_size}, actions={action_size}, "
            f"actor/critic={cfg.num_layers}x{cfg.hidden_units} ELU, envs={num_envs}, "
            f"time_scale={cfg.time_scale:g}, progress={cfg.progress_step_mode}, device={device}"
        )
        print(
            f"DDPG settings: actor_lr={cfg.actor_learning_rate:g}, "
            f"critic_lr={cfg.critic_learning_rate:g}, batch={cfg.batch_size}, "
            f"buffer={cfg.replay_buffer_size}, start_steps={cfg.start_steps}, "
            f"update_after={cfg.update_after}, update_every={cfg.update_every}"
        )
        progress_step = get_progress_step(cfg.progress_step_mode, decision_step, environment_step)
        write_training_status(
            run_output_dir,
            {
                "status": "running",
                "behavior_name": cfg.behavior_name,
                "run_id": cfg.run_id,
                "step": progress_step,
                "decision_steps": decision_step,
                "environment_steps": environment_step,
                "episodes": total_episodes,
                "num_envs": num_envs,
                "time_scale": cfg.time_scale,
                "progress_step_mode": cfg.progress_step_mode,
                "device": str(device),
            },
        )

        while get_progress_step(cfg.progress_step_mode, decision_step, environment_step) < cfg.max_steps:
            stepped_agents = 0
            for env_index, env in enumerate(envs):
                decision_steps, terminal_steps = env.get_steps(behavior_name)
                decision_obs = flatten_vector_obs(decision_steps.obs)
                actions = np.zeros((len(decision_steps), action_size), dtype=np.float32)

                for row, agent_id in enumerate(decision_steps.agent_id):
                    obs = decision_obs[row]
                    if decision_step < cfg.start_steps:
                        action = np.random.uniform(-1.0, 1.0, size=action_size).astype(np.float32)
                    else:
                        with torch.no_grad():
                            obs_tensor = torch.as_tensor(obs, device=device).unsqueeze(0)
                            action = actor(obs_tensor).cpu().numpy()[0]
                        action += np.random.normal(0.0, noise_scale(decision_step, cfg), size=action_size)
                        action = np.clip(action, -1.0, 1.0).astype(np.float32)

                    agent_key = (env_index, int(agent_id))
                    last_obs[agent_key] = obs
                    last_action[agent_key] = action
                    episode_lengths[agent_key] = episode_lengths.get(agent_key, 0)
                    actions[row] = action

                env.set_actions(behavior_name, ActionTuple(continuous=actions))
                stepped_agents += len(decision_steps)

            for env in envs:
                env.step()

            decision_step += stepped_agents
            environment_step += num_envs
            progress_step = get_progress_step(cfg.progress_step_mode, decision_step, environment_step)
            collect_environment_stats(stats_channels, pending_environment_stats)

            for env_index, env in enumerate(envs):
                next_decision_steps, next_terminal_steps = env.get_steps(behavior_name)
                next_decision_obs = flatten_vector_obs(next_decision_steps.obs)
                next_terminal_obs = flatten_vector_obs(next_terminal_steps.obs) if len(next_terminal_steps) else None

                for row, agent_id in enumerate(next_decision_steps.agent_id):
                    agent_key = (env_index, int(agent_id))
                    if agent_key in last_obs:
                        reward = float(next_decision_steps.reward[row])
                        replay.add(last_obs[agent_key], last_action[agent_key], reward, next_decision_obs[row], False)
                        episode_returns[agent_key] = episode_returns.get(agent_key, 0.0) + reward
                        episode_lengths[agent_key] = episode_lengths.get(agent_key, 0) + 1

                for row, agent_id in enumerate(next_terminal_steps.agent_id):
                    agent_key = (env_index, int(agent_id))
                    if agent_key in last_obs and next_terminal_obs is not None:
                        reward = float(next_terminal_steps.reward[row])
                        replay.add(last_obs[agent_key], last_action[agent_key], reward, next_terminal_obs[row], True)
                        episode_return = episode_returns.pop(agent_key, 0.0) + reward
                        episode_length = episode_lengths.pop(agent_key, 0) + 1
                        recent_returns.append(episode_return)
                        total_episodes += 1
                        summary_writer.add_scalar("Environment/Cumulative Reward", episode_return, progress_step)
                        summary_writer.add_scalar("Environment/Episode Length", episode_length, progress_step)
                        last_obs.pop(agent_key, None)
                        last_action.pop(agent_key, None)

            if replay.size >= cfg.update_after and decision_step > 0 and decision_step % cfg.update_every == 0:
                obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = replay.sample(cfg.batch_size, device)

                with torch.no_grad():
                    next_actions = actor_target(next_obs_batch)
                    target_q = critic_target(next_obs_batch, next_actions)
                    y = reward_batch + cfg.gamma * (1.0 - done_batch) * target_q

                q = critic(obs_batch, action_batch)
                critic_loss = F.mse_loss(q, y)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
                last_critic_loss = float(critic_loss.item())

                actor_actions = actor(obs_batch)
                actor_loss = -critic(obs_batch, actor_actions).mean()
                actor_loss = actor_loss + cfg.action_l2_weight * actor_actions.pow(2).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                last_actor_loss = float(actor_loss.item())

                soft_update(actor, actor_target, cfg.tau)
                soft_update(critic, critic_target, cfg.tau)

            if progress_step % cfg.summary_freq < max(num_envs, stepped_agents, 1):
                mean_return = float(np.mean(recent_returns)) if recent_returns else 0.0
                std_return = float(np.std(recent_returns)) if recent_returns else 0.0
                elapsed = max(time.time() - training_start_time, 1e-6)
                steps_per_second = progress_step / elapsed
                decision_steps_per_second = decision_step / elapsed
                print(
                    f"[INFO] {cfg.behavior_name}. "
                    f"Step: {progress_step}. "
                    f"Time Elapsed: {elapsed:.3f} s. "
                    f"Mean Reward: {mean_return:.3f}. "
                    f"Std of Reward: {std_return:.3f}. "
                    f"Training."
                )
                summary_writer.add_scalar("Policy/Recent Mean Reward", mean_return, progress_step)
                summary_writer.add_scalar("Policy/Recent Reward Std", std_return, progress_step)
                summary_writer.add_scalar("Policy/Noise Scale", noise_scale(decision_step, cfg), progress_step)
                summary_writer.add_scalar("Buffer/Replay Size", replay.size, progress_step)
                summary_writer.add_scalar("Timing/Steps Per Second", steps_per_second, progress_step)
                summary_writer.add_scalar("Timing/Decision Steps Per Second", decision_steps_per_second, progress_step)
                summary_writer.add_scalar("Training/Decision Steps", decision_step, progress_step)
                summary_writer.add_scalar("Training/Environment Steps", environment_step, progress_step)
                summary_writer.add_scalar("Training/Episodes", total_episodes, progress_step)
                if last_actor_loss is not None:
                    summary_writer.add_scalar("Losses/Actor", last_actor_loss, progress_step)
                if last_critic_loss is not None:
                    summary_writer.add_scalar("Losses/Critic", last_critic_loss, progress_step)
                if recent_returns and mean_return > best_recent_mean_return:
                    best_recent_mean_return = mean_return
                    summary_writer.add_scalar("Policy/Best Recent Mean Reward", best_recent_mean_return, progress_step)
                    save_checkpoint(
                        artifact_output_dir / "best",
                        actor,
                        critic,
                        actor_target,
                        critic_target,
                        actor_optimizer,
                        critic_optimizer,
                        progress_step,
                        obs_size,
                        action_size,
                        cfg,
                    )
                write_environment_stats(summary_writer, pending_environment_stats, progress_step)
                summary_writer.flush()
                write_training_status(
                    run_output_dir,
                    {
                        "status": "running",
                        "behavior_name": cfg.behavior_name,
                        "run_id": cfg.run_id,
                        "step": progress_step,
                        "decision_steps": decision_step,
                        "environment_steps": environment_step,
                        "episodes": total_episodes,
                        "recent_mean_reward": mean_return,
                        "recent_std_reward": std_return,
                        "noise_scale": noise_scale(decision_step, cfg),
                        "replay_size": replay.size,
                        "steps_per_second": steps_per_second,
                        "decision_steps_per_second": decision_steps_per_second,
                        "actor_loss": last_actor_loss,
                        "critic_loss": last_critic_loss,
                    },
                )

            if progress_step % cfg.save_freq < max(num_envs, stepped_agents, 1):
                save_checkpoint(
                    artifact_output_dir,
                    actor,
                    critic,
                    actor_target,
                    critic_target,
                    actor_optimizer,
                    critic_optimizer,
                    progress_step,
                    obs_size,
                    action_size,
                    cfg,
                )

        save_checkpoint(
            artifact_output_dir,
            actor,
            critic,
            actor_target,
            critic_target,
            actor_optimizer,
            critic_optimizer,
            get_progress_step(cfg.progress_step_mode, decision_step, environment_step),
            obs_size,
            action_size,
            cfg,
        )
        progress_step = get_progress_step(cfg.progress_step_mode, decision_step, environment_step)
        collect_environment_stats(stats_channels, pending_environment_stats)
        write_environment_stats(summary_writer, pending_environment_stats, progress_step)
        summary_writer.flush()
        write_training_status(
            run_output_dir,
            {
                "status": "completed",
                "behavior_name": cfg.behavior_name,
                "run_id": cfg.run_id,
                "step": progress_step,
                "decision_steps": decision_step,
                "environment_steps": environment_step,
                "episodes": total_episodes,
                "recent_mean_reward": float(np.mean(recent_returns)) if recent_returns else 0.0,
                "recent_std_reward": float(np.std(recent_returns)) if recent_returns else 0.0,
                "actor_loss": last_actor_loss,
                "critic_loss": last_critic_loss,
            },
        )
    finally:
        summary_writer.close()
        for env in envs:
            env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DDPG-FC-350-E-PID for the Unity ball-and-plate project.")
    parser.add_argument("--env", default=None, help="Path to a Unity executable. Omit this and press Play in the Editor.")
    parser.add_argument("--config", default="Tools/DDPGFC350EPid/ddpg_fc_350_e_pid.yaml")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--run-id", default=None, help="Override the run_id from the config.")
    parser.add_argument("--seed", type=int, default=None, help="Override the random seed from the config.")
    parser.add_argument("--num-envs", type=int, default=None, help="Number of Unity environment instances to launch.")
    parser.add_argument("--time-scale", type=float, default=None, help="Override Unity engine time scale.")
    parser.add_argument("--force", action="store_true", help="Overwrite the existing results directory for this run_id.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even when CUDA is available.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

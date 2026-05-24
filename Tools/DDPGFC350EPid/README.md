# DDPG-FC-350-E-PID

This adds a separate DDPG workflow for the new Unity behavior `ddpg_fc_350_e_pid_agent`.

The design mirrors the referenced paper:

- DDPG actor and critic networks
- Two hidden fully connected layers
- 350 nodes per hidden layer
- ELU activation throughout the hidden layers
- Continuous actions that represent PID gains
- E-PID observations based on error, integral error, and derivative error

## Unity setup

Add the new `DDPGFC350EPidAgent` component to the same GameObject currently used for PID tuning, or duplicate the PPO training object/prefab and replace only the Agent script on the duplicate.

The new script configures:

- Behavior name: `ddpg_fc_350_e_pid_agent`
- Vector observations: 10
- Continuous actions: 3

## Training

From the project root, using the config shaped like ML-Agents trainer YAML:

```powershell
venv_mlagents23\Scripts\python.exe Tools\DDPGFC350EPid\ddpg_fc_350_e_pid_train.py --config Tools\DDPGFC350EPid\config.yaml --env Builds\BallBalancer_Kinematic.exe --run-id DDPG_08 --num-envs 5 --time-scale 20 --force
```

To connect to the Unity Editor instead of a build, omit `--env`, start the command, then press Play in Unity:

```powershell
venv_mlagents23\Scripts\python.exe Tools\DDPGFC350EPid\ddpg_fc_350_e_pid_train.py --config Tools\DDPGFC350EPid\config.yaml
```

The code is also split into a PPO/SAC-like trainer folder:

```text
Tools/DDPGFC350EPid/ddpg_fc_350_e_pid/
  __init__.py
  optimizer_torch.py
  trainer.py
```

Checkpoints and the exported actor ONNX model are written under:

```text
results/<run_id>/ddpg_fc_350_e_pid_agent/
```

The custom DDPG runner now sends Unity engine settings through the same
`EngineConfigurationChannel` path used by `mlagents-learn`, including
`time_scale`, resolution, quality, target frame rate, and capture frame rate.
The TensorBoard x-axis uses environment progress by default so it is easier to
compare with PPO runs. Decision/training progress is still logged separately as
`Training/Decision Steps`.

Unity ML-Agents in this project only provides built-in PPO and SAC trainers, so this script is the DDPG replacement path rather than a `trainer_type: ddpg` YAML for `mlagents-learn`.

## ML-Agents Learn Mode

Unity ML-Agents does not include a built-in DDPG trainer in this project version. If you want the same runner UX as PPO, use the SAC config. It is not DDPG, but it is an off-policy continuous-control actor-critic trainer and works with `mlagents-learn`.

```powershell
venv_mlagents23\Scripts\mlagents-learn.exe Tools\DDPGFC350EPid\mlagents_sac_fc350_epid.yaml --env Builds\BallBalancer_Kinematic.exe --run-id sac_fc350_e_pid_01 --num-envs 4 --time-scale 20 --no-graphics
```

Or use the helper script:

```powershell
.\Tools\DDPGFC350EPid\run_mlagents_sac_fc350_epid.ps1 -NumEnvs 4 -TimeScale 20 -RunId sac_fc350_e_pid_01
```

If `Ctrl+C` does not stop a stuck run:

```powershell
.\Tools\DDPGFC350EPid\stop_training_processes.ps1
```

## Multiple Environments

Run multiple Unity executable instances with `--num-envs`. Each instance uses `worker_id + index`, so `--num-envs 4` with `worker_id: 0` launches workers `0, 1, 2, 3`.

```powershell
venv_mlagents23\Scripts\python.exe Tools\DDPGFC350EPid\ddpg_fc_350_e_pid_train.py --config Tools\DDPGFC350EPid\config.yaml --env Builds\BallBalancer_Kinematic.exe --run-id DDPG_09 --num-envs 4 --time-scale 20 --force
```

Use the Unity Editor only for one environment. For multiple environments, use a built `.exe`.

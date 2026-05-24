param(
    [int]$NumEnvs = 4,
    [int]$TimeScale = 20,
    [string]$RunId = "sac_fc350_e_pid_01",
    [string]$EnvPath = "Builds\BallBalancer_Kinematic.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$ConfigPath = Join-Path $PSScriptRoot "mlagents_sac_fc350_epid.yaml"
$Python = Join-Path $ProjectRoot "venv_mlagents23\Scripts\python.exe"

Set-Location $ProjectRoot

& $Python -m mlagents.trainers.learn $ConfigPath `
    --env $EnvPath `
    --run-id $RunId `
    --num-envs $NumEnvs `
    --time-scale $TimeScale `
    --no-graphics `
    --width 84 `
    --height 84

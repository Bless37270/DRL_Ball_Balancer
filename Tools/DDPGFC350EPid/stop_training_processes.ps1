$ErrorActionPreference = "SilentlyContinue"

Get-Process python, pythonw, BallBalancer_Kinematic, UnityCrashHandler64 |
    Where-Object {
        $_.ProcessName -eq "BallBalancer_Kinematic" -or
        $_.Path -like "*BallBalancer_Kinematic*" -or
        $_.Path -like "*venv_mlagents23*"
    } |
    Stop-Process -Force

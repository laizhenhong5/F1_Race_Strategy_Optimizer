# Virtual Environment Activation

Use the workspace virtual environment whenever you work on this project.

## Activate in PowerShell

Open a PowerShell terminal in the project root and run:

```powershell
. .\.venv\Scripts\Activate.ps1
```

If PowerShell blocks scripts, allow activation for the session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
. .\.venv\Scripts\Activate.ps1
```

## Activate in Command Prompt (cmd)

```cmd
.venv\Scripts\activate.bat
```

## Activate in Git Bash / WSL

```bash
source .venv/Scripts/activate
```

## Verify activation

```bash
python --version
python -c "import fastf1, pandas, numpy, matplotlib, sklearn"
```

## Install dependencies

If you need to reinstall or update dependencies, run:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

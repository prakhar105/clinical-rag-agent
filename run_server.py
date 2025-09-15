# run_server.py
import subprocess
import os

# Path to activate script
venv_activate = os.path.join(".venv", "Scripts", "activate.bat")

# Run the activation + run script in the same shell
command = f'cmd /k "call {venv_activate} && python run.py"'
subprocess.run(command, shell=True)

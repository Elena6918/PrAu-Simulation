import subprocess
import sys
import os

# Function to create a virtual environment
def create_virtualenv(venv_dir):
    subprocess.check_call([sys.executable, "-m", "venv", venv_dir])

# Function to install packages from requirements.txt in the virtual environment
def install_packages(venv_dir):
    pip_executable = os.path.join(venv_dir, 'bin', 'pip') if os.name != 'nt' else os.path.join(venv_dir, 'Scripts', 'pip')
    subprocess.check_call([pip_executable, "install", "-r", "requirements.txt"])

# Function to run a Python script using the virtual environment's Python interpreter
def run_script(venv_dir, script_name):
    python_executable = os.path.join(venv_dir, 'bin', 'python') if os.name != 'nt' else os.path.join(venv_dir, 'Scripts', 'python')
    subprocess.check_call([python_executable, script_name])

if __name__ == "__main__":
    venv_dir = "venv"

    # Create virtual environment
    print("Creating virtual environment...")
    create_virtualenv(venv_dir)

    # Install required packages
    print("Installing packages in virtual environment...")
    install_packages(venv_dir)

    # List of scripts to execute
    scripts_to_run = [
        "scenario_2_accuracy.py",
        "plot_fpr.py",
        "simulate_colluder_num.py"
        # add other scripts as needed
    ]

    # Execute each script
    for script in scripts_to_run:
        print(f"Running {script} in virtual environment...")
        run_script(venv_dir, script)

    print("All scripts executed successfully.")

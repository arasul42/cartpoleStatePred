import subprocess
import os

def run_docker_compose():
    """Run the 'docker compose up' command."""
    try:
        subprocess.run(["docker", "compose", "up", "-d"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running docker compose: {e}")
    except FileNotFoundError:
        print("Docker is not installed or not found in PATH.")

def allow_xhost_access():
    """Allow local Docker access to X11."""
    try:
        os.system("xhost +local:docker")
    except Exception as e:
        print(f"Error occurred while allowing X11 access: {e}")

def revoke_xhost_access():
    """Revoke local Docker access to X11."""
    try:
        os.system("xhost -local:docker")
    except Exception as e:
        print(f"Error occurred while revoking X11 access: {e}")

def run_docker_and_execute_script():
    """Executes cart_pole.py inside the container and keeps the shell open."""
    try:
        subprocess.run([
            "docker", "exec", "-it", "ubuntu_gym",
            "bash", "-c", "python3 cart_pole.py; exec bash"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running script inside container: {e}")



if __name__ == "__main__":
    run_docker_compose()
    allow_xhost_access()
    try:
        run_docker_and_execute_script()
    finally:
        revoke_xhost_access()
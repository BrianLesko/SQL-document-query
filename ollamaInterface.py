# Brian Lesko
# This app runs a ollama docker image intended for use with the Mistral 7B LLM model

import subprocess
import os
import pexpect

class ollamaInterface:
    def __init__(self):
        self.port = '11434'
        self.model_dir = './models'  # The directory where the models are stored
        self.model = 'Mistral-7B-v0.2-Instruct.tar'  # The model to use
        self.model_path = os.path.join(self.model_dir, self.model)
        # Docker container parameters
        self.name = 'ollama'

    def start_container(self):
        if self.container_is_running():
            return f"Container {self.name} is already running."
        elif self.container_exists():
            subprocess.run(f"docker start {self.name}", shell=True, check=True)
            return f"Container {self.name} started."
        else:
            return f'Container {self.name} does not exist. Please build the container image with docker pull and then run it with: "docker run -d -v ./models:/root/.ollama -p 11434:11434 --name ollama ollama/ollama" '

    def container_exists(self):
        result = subprocess.run(f"docker ps -a --filter name={self.name}", shell=True, capture_output=True, text=True)
        return self.name in result.stdout

    def container_is_running(self):
        result = subprocess.run(f"docker ps --filter name={self.name}", shell=True, capture_output=True, text=True)
        return self.name in result.stdout
    
    def stop_container(self):
        subprocess.run(f"docker stop {self.name}", shell=True, check=True)
        return f"Container {self.name} stopped"

    def start(self):
        start_message = self.start_container()
        if "started" in start_message or "already running" in start_message:
            command = f"docker exec -it ollama ollama run mistral" # {self.model}
            self.process = pexpect.spawn(command, encoding='utf-8')
        else:
            raise Exception(start_message)

    def is_container_running(self):
        try:
            return self.name in subprocess.run(f"docker ps --filter name={self.name} --format '{{{{.Names}}}}'", shell=True, check=True, stdout=subprocess.PIPE, universal_newlines=True).stdout.strip().split('\n')
        except subprocess.CalledProcessError:
            return False
        
    def is_docker_running(self):
        try:
            subprocess.run("docker info", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            return True 
        except subprocess.CalledProcessError:
            return False
        
    def get_docker_version(self):
        try:
            return subprocess.run("docker --version", shell=True, check=True, stdout=subprocess.PIPE, universal_newlines=True).stdout.strip()
        except subprocess.CalledProcessError:
            return "Docker not installed"

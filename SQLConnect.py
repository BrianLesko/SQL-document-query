# Brian Lesko
# This app connects to a MySQL server

import pymysql
import subprocess

class SQLConnectDocker:
    def __init__(self):
        # SQL connection parameters
        self.server = '127.0.0.1'  # Docker's mapped MySQL port on localhost
        self.port = 3306
        self.username = 'root'  # Default MySQL root user
        self.password = 'lesko'  # The password you set for MySQL
        self.database = 'mysql'  # Default database, change as needed
        # Docker container parameters
        self.name = 'my-mysql'
        # Docker status
        self.docker_version = self.get_docker_version()
        self.docker_is_running = self.is_docker_running()
        self.container_is_running = self.is_container_running()
        if self.docker_is_running and not self.container_is_running:
            print("Docker is running, but the container is not. Starting the container...")
            self.start_container()

    def is_container_running(self):
        try:
            return self.name in subprocess.run(f"docker ps --filter name={self.name} --format '{{{{.Names}}}}'", shell=True, check=True, stdout=subprocess.PIPE, universal_newlines=True).stdout.strip().split('\n')
        except subprocess.CalledProcessError:
            return False
        
    def is_docker_running(self):
        try:
            # Run a simple Docker command
            subprocess.run("docker info", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            return True  # The command succeeded, so Docker is running
        except subprocess.CalledProcessError:
            return False  # The command failed, Docker daemon might not be running
        
    def get_docker_version(self):
        try:
            return subprocess.run("docker --version", shell=True, check=True, stdout=subprocess.PIPE, universal_newlines=True).stdout.strip()
        except subprocess.CalledProcessError:
            return "Docker not installed"

    def rebuild_container(self):
        subprocess.run(f"docker stop {self.name}", shell=True, check=True)
        subprocess.run(f"docker rm {self.name}", shell=True, check=True)
        subprocess.run(f"docker run --name {self.name} -e MYSQL_ROOT_PASSWORD={self.password} -d -p 3306:3306 mysql", shell=True, check=True)
        return f"Container {self.name} rebuilt"

    def start_container(self):
        subprocess.run(f"docker start {self.name}", shell=True, check=True)
        return f"Container {self.name} started"
    
    def stop_container(self):
        subprocess.run(f"docker stop {self.name}", shell=True, check=True)
        return f"Container {self.name} stopped"

    def connect(self):
            self.conn = pymysql.connect(
                host=self.server,
                port=self.port,
                user=self.username,
                password=self.password,
                db=self.database,
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
            self.connection = self.conn
            self.cursor = self.conn.cursor()
            return self.cursor
    
    def get_summary(self):
        self.cursor.execute("SHOW DATABASES;")
        return self.cursor.fetchall()
    
    def get_tables(self):
        self.cursor.execute("USE user")   
        self.cursor.execute("SHOW TABLES")
        return self.cursor.fetchall()

    def query(self, query):
        try:
            self.cursor.execute(query)
            return self.cursor.fetchall() 
        except Exception as e:
            print("An error occured during the query")
            return e

    def update(self, sql):
        try:
            self.cursor.execute(sql)
            self.connection.commit()
        except Exception as e:
            print(f"An error occurred during the update: {str(e)}")
    
    def close(self):
        self.conn.close()

    def write_to_table(self, filename, Characters, UTF8_length, compressed_length):
        try:
            table = 'size'
            self.query("USE user")
            self.update(f"INSERT INTO {table} (name, characters, utf8_size, compressed_size) VALUES ('{filename}', {Characters}, {UTF8_length}, {compressed_length});")
            self.connection.commit()
        except Exception as e:
            print(f"An error occurred during the query: {str(e)}")

    def describe_table(self, table_name):
        try:
            self.cursor.execute(f"DESCRIBE {table_name};")
            return self.cursor.fetchall()
        except Exception as e:
            return e

def example():
    # Example usage
    sql = SQLConnectClass()
    error = sql.connect()
    if error:
        print(f"Connection error: {error}")
    else:
        result = sql.query("SHOW DATABASES;")
        print(result)
        sql.close()

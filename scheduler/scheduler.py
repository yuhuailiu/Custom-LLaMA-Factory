import socket
import threading
import queue
import subprocess
import time
import os
import argparse
import logging
from datetime import datetime
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

HOST = '127.0.0.1'
PORT = 9999
LOG_FILE = os.path.join(SCRIPT_DIR, 'scheduler.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

class JobScheduler:
    def __init__(self):
        self.job_queue = queue.Queue()
        self.is_running = True
        self.current_job = None
        # --- New attributes ---
        self.current_process = None
        self.terminate_requested = False
        self.termination_delay = 5  # seconds
        self.lock = threading.Lock()

    def log_stream(self, stream, log_level):
        for line in iter(stream.readline, ''):
            logging.log(log_level, line.strip())
        stream.close()

    def terminate_current_job(self):
        """Send terminate signal to the currently running subprocess and mark for delayed wait."""
        with self.lock:
            if not self.current_process:
                return False
            try:
                self.current_process.terminate()
                self.terminate_requested = True
                logging.info("✅ 已发送终止信号给当前作业。")
                return True
            except Exception as e:
                logging.error(f"❌ 终止当前作业时出错: {e}")
                return False

    def worker_thread(self):
        logging.info("Worker thread started. Waiting for jobs...")
        while self.is_running:
            try:
                script_path = self.job_queue.get(timeout=1)
                with self.lock:
                    self.current_job = script_path

                logging.info(f"--- Starting new job: {script_path} ---")
                start_time = time.time()

                try:
                    process = subprocess.Popen(
                        ['bash', script_path],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        encoding='utf-8',
                        errors='replace',
                        bufsize=1
                    )
                    # --- Record child process handle ---
                    with self.lock:
                        self.current_process = process

                    stdout_thread = threading.Thread(
                        target=self.log_stream,
                        args=(process.stdout, logging.INFO)
                    )
                    stderr_thread = threading.Thread(
                        target=self.log_stream,
                        args=(process.stderr, logging.ERROR)
                    )
                    stdout_thread.start()
                    stderr_thread.start()

                    return_code = process.wait()
                    stdout_thread.join()
                    stderr_thread.join()
                    end_time = time.time()
                    total_time = int(end_time - start_time)

                    if return_code == 0:
                        logging.info(f"--- Job '{script_path}' completed successfully in {total_time} seconds. ---")
                    else:
                        logging.error(f"--- Job '{script_path}' failed with return code {return_code} after {total_time} seconds. ---")

                except Exception as e:
                    logging.error(f"An exception occurred while running job '{script_path}': {e}")

                finally:
                    # If terminated by terminate request, wait for a delay
                    if self.terminate_requested:
                        logging.info(f"Waiting {self.termination_delay} seconds to ensure job is fully closed...")
                        time.sleep(self.termination_delay)
                        self.terminate_requested = False

                    with self.lock:
                        self.current_job = None
                        self.current_process = None
                    self.job_queue.task_done()

            except queue.Empty:
                continue

    def server_thread(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((HOST, PORT))
                s.listen()
                logging.info(f"Scheduler service listening on {HOST}:{PORT}")
                while self.is_running:
                    s.settimeout(1.0)
                    try:
                        conn, addr = s.accept()
                    except socket.timeout:
                        continue

                    with conn:
                        data = conn.recv(1024).decode('utf-8').strip()
                        if not data:
                            continue

                        if data == '__STATUS__':
                            with self.lock:
                                curr = self.current_job
                            q_size = self.job_queue.qsize()
                            status_msg = f"Currently Running: {curr}\nJobs in Queue: {q_size}"
                            conn.sendall(status_msg.encode('utf-8'))

                        # --- Added terminate command ---
                        elif data == '__TERMINATE__':
                            if self.terminate_current_job():
                                msg = "✅ Termination signal sent to current job."
                            else:
                                msg = "⚠️ No job is currently running."
                            conn.sendall(msg.encode('utf-8'))

                        else:
                            # Submit new job
                            script_path = data
                            if os.path.exists(script_path) and script_path.endswith('.sh'):
                                self.job_queue.put(script_path)
                                msg = f"✅ Job '{script_path}' submitted successfully. It's position #{self.job_queue.qsize()} in the queue."
                                logging.info(msg)
                                conn.sendall(msg.encode('utf-8'))
                            else:
                                msg = f"❌ Error: Script '{script_path}' not found or is not a .sh file."
                                logging.warning(msg)
                                conn.sendall(msg.encode('utf-8'))

            except OSError as e:
                logging.error(f"Could not start server on {HOST}:{PORT}. Is another instance running? Error: {e}")
            finally:
                logging.info("Server thread shutting down.")

    def start(self):
        self.worker = threading.Thread(target=self.worker_thread, daemon=True)
        self.server = threading.Thread(target=self.server_thread, daemon=True)
        self.worker.start()
        self.server.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("Shutdown signal received. Shutting down gracefully...")
            self.is_running = False

# === Client 部分：增加 kill 命令 ===
def submit_job(script_path):
    abs_path = os.path.abspath(script_path)
    if not os.path.exists(abs_path) or not abs_path.endswith('.sh'):
        print(f"❌ Error: Script '{abs_path}' not found or is not a .sh file.")
        return
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall(abs_path.encode('utf-8'))
            print(s.recv(1024).decode('utf-8'))
    except ConnectionRefusedError:
        print(f"❌ Error: Could not connect to the scheduler service on {HOST}:{PORT}. Is it running?")

def get_status():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall(b'__STATUS__')
            print("--- Scheduler Status ---")
            print(s.recv(1024).decode('utf-8'))
            print("------------------------")
    except ConnectionRefusedError:
        print(f"❌ Error: Could not connect to the scheduler service on {HOST}:{PORT}. Is it running?")

def kill_current():
    """Send terminate instruction to the currently running job."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall(b'__TERMINATE__')
            print(s.recv(1024).decode('utf-8'))
    except ConnectionRefusedError:
        print(f"❌ Error: Could not connect to the scheduler service on {HOST}:{PORT}. Is it running?")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A simple sequential job scheduler for shell scripts.")
    subparsers = parser.add_subparsers(dest='command', required=True)
    subparsers.add_parser('start', help='Start the scheduler service in the foreground.')
    submit_parser = subparsers.add_parser('submit', help='Submit a shell script to the queue.')
    submit_parser.add_argument('script_path', type=str, help='The path to the .sh script to execute.')
    subparsers.add_parser('status', help='Check the current status of the scheduler.')
    subparsers.add_parser('kill', help='Terminate the currently running job.')  # Added kill

    args = parser.parse_args()
    if args.command == 'start':
        scheduler = JobScheduler()
        scheduler.start()
    elif args.command == 'submit':
        submit_job(args.script_path)
    elif args.command == 'status':
        get_status()
    elif args.command == 'kill':
        kill_current()

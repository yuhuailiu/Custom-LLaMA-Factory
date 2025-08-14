import socket
import threading
import queue
import subprocess
import time
import os
import argparse
import logging
import shutil
import json
import re
from datetime import datetime
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

HOST = '127.0.0.1'
PORT = 9999
LOG_FILE = os.path.join(SCRIPT_DIR, 'scheduler.log')
# Merge settings
MERGE_SCRIPT = "/data2/lyh/Custom-LLaMA-Factory/scripts/merge_lora_from_yaml.py"
CONDA_ENV_FOR_MERGE = os.getenv("SCHEDULER_CONDA_ENV", "lyh-lf")
 # Eval generation and collection
GEN_EVAL_SCRIPT = "/data2/lyh/Custom-LLaMA-Factory/scripts/gen_eval_from_train_yaml.py"
COLLECT_EVAL_SCRIPT = "/data2/lyh/Custom-LLaMA-Factory/scripts/collect_eval_results.py"
EVAL_RESULTS_ROOT = "/data2/lyh/eval_results"

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

    def log_stream(self, stream, log_level, file_handle=None):
        for line in iter(stream.readline, ''):
            logging.log(log_level, line.strip())
            if file_handle is not None:
                try:
                    file_handle.write(line)
                    file_handle.flush()
                except Exception:
                    pass
        stream.close()

    @staticmethod
    def _parse_yaml_output_dir(yaml_path: str) -> str:
        """Lightweight reader to extract output_dir from a YAML file without full dependency requirements."""
        try:
            import yaml  # type: ignore
        except Exception:
            yaml = None

        if yaml is not None:
            try:
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                if isinstance(data, dict) and 'output_dir' in data:
                    return str(data['output_dir'])
            except Exception:
                pass

        # Fallback: manual scan
        with open(yaml_path, 'r', encoding='utf-8') as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('output_dir:'):
                    return line.split(':', 1)[1].strip().strip("'\"")
        raise ValueError(f"Cannot find output_dir in YAML: {yaml_path}")

    @staticmethod
    def _parse_yaml_do_train(yaml_path: str):
        """Return the boolean of do_train in YAML if present, else None."""
        try:
            import yaml  # type: ignore
        except Exception:
            yaml = None

        if yaml is not None:
            try:
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                if isinstance(data, dict) and 'do_train' in data:
                    val = data['do_train']
                    if isinstance(val, bool):
                        return val
            except Exception:
                pass
        # Fallback: manual scan
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith('#'):
                        continue
                    if line.startswith('do_train:'):
                        token = line.split(':', 1)[1].strip().strip('\'"')
                        if token.lower() in {'true', '1', 'yes', 'y'}:
                            return True
                        if token.lower() in {'false', '0', 'no', 'n'}:
                            return False
                        break
        except Exception:
            pass
        return None

    @staticmethod
    def _parse_yaml_command(task_str: str) -> tuple[str, str]:
        """Parse incoming task string for command and yaml path.

        Accepted formats:
        - "path/to/config.yaml" -> assumes command 'train'
        - "train:/abs/path.yaml"
        - "eval:/abs/path.yaml"
        """
        task = task_str.strip()
        if task.lower().endswith('.yaml'):
            return 'train', task
        if ':' in task:
            prefix, path = task.split(':', 1)
            prefix = prefix.strip().lower()
            path = path.strip()
            if prefix in {'train', 'eval'}:
                return prefix, path
        raise ValueError("Unrecognized YAML task format. Use '/path/to.yaml' or 'train:/path/to.yaml' or 'eval:/path/to.yaml'.")

    @staticmethod
    def _build_job_env(output_dir: str) -> dict:
        """Build environment for llamafactory CLI to mirror our .sh runs.

        - FORCE_TORCHRUN=1
        - ASCEND_RT_VISIBLE_DEVICES: inherit or default to 0,1,2,3,4,5,6,7
        - SWANLAB_MODE: inherit or default to disabled
        """
        env = os.environ.copy()
        env.setdefault("FORCE_TORCHRUN", "1")
        env.setdefault("ASCEND_RT_VISIBLE_DEVICES", os.getenv("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7"))
        env.setdefault("SWANLAB_MODE", os.getenv("SWANLAB_MODE", "disabled"))
        # optional: expose workdir-like hints
        env.setdefault("LLAMABOARD_WORKDIR", output_dir)
        return env

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
                    # Keep context for optional merge
                    ran_yaml_task = False
                    ran_yaml_command = None  # 'train' or 'eval'
                    ran_yaml_path = None
                    ran_yaml_out_dir = None
                    # Determine job type (.sh vs. YAML task) with robust parse-first approach
                    log_file_handle = None
                    yaml_do_train = None  # type: ignore[var-annotated]
                    try:
                        # Attempt to parse as YAML task (supports 'train:/path.yaml', 'eval:/path.yaml', or '/path.yaml')
                        command, yaml_path = self._parse_yaml_command(str(script_path).strip())
                        is_yaml_task = True
                        try:
                            yaml_do_train = self._parse_yaml_do_train(yaml_path)
                        except Exception:
                            yaml_do_train = None
                    except Exception:
                        is_yaml_task = False

                    if is_yaml_task:
                        ran_yaml_task = True
                        ran_yaml_command = command
                        ran_yaml_path = yaml_path
                        out_dir = self._parse_yaml_output_dir(yaml_path)
                        os.makedirs(out_dir, exist_ok=True)
                        ran_yaml_out_dir = out_dir
                        # Copy YAML into output_dir with timestamp to avoid overwrites
                        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                        base = os.path.basename(yaml_path)
                        name, ext = os.path.splitext(base)
                        copied_yaml = os.path.join(out_dir, f"{name}_{ts}{ext}")
                        try:
                            shutil.copy2(yaml_path, copied_yaml)
                            logging.info(f"Copied YAML to {copied_yaml}")
                        except Exception as e:
                            logging.warning(f"Failed to copy YAML to output_dir: {e}")

                        # Open log file in output_dir
                        log_path = os.path.join(out_dir, f"llamafactory_{command}_{ts}.log")
                        try:
                            log_file_handle = open(log_path, 'a', encoding='utf-8')
                            logging.info(f"Streaming logs to {log_path}")
                        except Exception as e:
                            logging.warning(f"Failed to open log file {log_path}: {e}")

                        # Launch llamafactory-cli
                        process = subprocess.Popen(
                            ['llamafactory-cli', command, yaml_path],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            encoding='utf-8',
                            errors='replace',
                            bufsize=1,
                            env=self._build_job_env(out_dir),
                        )
                    else:
                        # Fallback: treat as .sh job; Mirror env for consistency as well
                        process = subprocess.Popen(
                            ['bash', str(script_path).strip()],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            encoding='utf-8',
                            errors='replace',
                            bufsize=1,
                            env=self._build_job_env(os.getcwd()),
                        )
                    # --- Record child process handle ---
                    with self.lock:
                        self.current_process = process

                    stdout_thread = threading.Thread(
                        target=self.log_stream,
                        args=(process.stdout, logging.INFO, log_file_handle)
                    )
                    stderr_thread = threading.Thread(
                        target=self.log_stream,
                        args=(process.stderr, logging.ERROR, log_file_handle)
                    )
                    stdout_thread.start()
                    stderr_thread.start()

                    return_code = process.wait()
                    stdout_thread.join()
                    stderr_thread.join()
                    if log_file_handle is not None:
                        try:
                            log_file_handle.close()
                        except Exception:
                            pass
                    end_time = time.time()
                    total_time = int(end_time - start_time)

                    if return_code == 0:
                        logging.info(f"--- Job '{script_path}' completed successfully in {total_time} seconds. ---")
                    else:
                        logging.error(f"--- Job '{script_path}' failed with return code {return_code} after {total_time} seconds. ---")

                    # Decide what just ran based on do_train flag; fallback to prefix/heuristic if missing
                    if ran_yaml_task and ran_yaml_path:
                        is_train_run = False
                        is_eval_run = False
                        if yaml_do_train is True:
                            is_train_run = True
                        elif yaml_do_train is False:
                            is_eval_run = True
                        else:
                            # Fallback: use command and filename heuristics
                            if ran_yaml_command == 'eval' or self._is_eval_yaml(ran_yaml_path):
                                is_eval_run = True
                            else:
                                is_train_run = True

                        if is_train_run and ran_yaml_out_dir:
                            # After training: merge (for lora) and enqueue eval
                            try:
                                self._run_merge_blocking(ran_yaml_path, ran_yaml_out_dir)
                            except Exception as me:
                                logging.warning(f"Merge step raised exception for {ran_yaml_path}: {me}")

                            try:
                                eval_yaml_path = self._gen_eval_yaml_blocking(ran_yaml_path)
                                if eval_yaml_path:
                                    logging.info(f"Enqueue eval (front): {eval_yaml_path}")
                                    # Use 'train' command to run evaluation YAML via training pipeline (do_train: false, do_eval: true)
                                    self._put_front(f"{eval_yaml_path}")
                                else:
                                    logging.warning("Eval YAML generation returned empty path; skipping enqueue.")
                            except Exception as ge:
                                logging.warning(f"Eval YAML generation failed for {ran_yaml_path}: {ge}")

                        if is_eval_run:
                            # After evaluation: refresh CSV summary
                            try:
                                self._collect_eval_results_blocking()
                            except Exception as ce:
                                logging.warning(f"Collect eval results failed: {ce}")

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

    def _run_merge_blocking(self, yaml_path: str, out_dir: str) -> None:
        """Run LoRA merge synchronously; failures do not block queue progression beyond this step.

        Command: conda run -n <env> python MERGE_SCRIPT --yaml <yaml_path> --clean_checkpoints
        """
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = os.path.join(out_dir, f"merge_lora_{ts}.log")
        logging.info(f"Starting merge for {yaml_path}. Logs: {log_path}")

        try:
            log_file = open(log_path, 'a', encoding='utf-8')
        except Exception as e:
            logging.warning(f"Cannot open merge log file {log_path}: {e}")
            log_file = None

        cmd = [
            'conda', 'run', '-n', CONDA_ENV_FOR_MERGE,
            'python', MERGE_SCRIPT, '--yaml', yaml_path, '--clean_checkpoints'
        ]

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
            )
            t_out = threading.Thread(target=self.log_stream, args=(proc.stdout, logging.INFO, log_file))
            t_err = threading.Thread(target=self.log_stream, args=(proc.stderr, logging.ERROR, log_file))
            t_out.start(); t_err.start()
            rc = proc.wait()
            t_out.join(); t_err.join()
            if rc == 0:
                logging.info(f"Merge finished successfully for {yaml_path}.")
            else:
                logging.warning(f"Merge exited with code {rc} for {yaml_path} (continuing queue).")
        except Exception as e:
            logging.warning(f"Merge process failed to start or crashed for {yaml_path}: {e}")
        finally:
            if log_file is not None:
                try:
                    log_file.close()
                except Exception:
                    pass

    def _run_python_in_env(self, args: list[str]) -> tuple[int, str, str]:
        """Run a python command inside the configured conda env and capture output."""
        cmd = ['conda', 'run', '-n', CONDA_ENV_FOR_MERGE, 'python'] + args
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
        )
        out, err = proc.communicate()
        return proc.returncode, out, err

    def _gen_eval_yaml_blocking(self, train_yaml_path: str) -> str:
        """Generate eval YAML using our helper script and return its absolute path."""
        rc, out, err = self._run_python_in_env([GEN_EVAL_SCRIPT, '--train_yaml', train_yaml_path])
        if rc != 0:
            raise RuntimeError(f"gen_eval_from_train_yaml failed: rc={rc}, stderr={err.strip()}")
        # Extract JSON summary printed by the script; parse last JSON object
        generated = ""
        try:
            # try fast path
            summary = json.loads(out)
            generated = summary.get('generated_eval_yaml', '')
        except Exception:
            try:
                # fallback: find last JSON object in the text
                matches = re.findall(r"\{[\s\S]*?\}\s*$", out, flags=re.MULTILINE)
                if matches:
                    summary = json.loads(matches[-1])
                    generated = summary.get('generated_eval_yaml', '')
            except Exception:
                pass
        if not generated:
            logging.debug(f"gen_eval stdout:\n{out}")
            logging.debug(f"gen_eval stderr:\n{err}")
            raise RuntimeError("Could not parse generated eval YAML path from script output")
        return generated

    def _collect_eval_results_blocking(self) -> None:
        """Run the results collector to refresh the summary CSV."""
        rc, out, err = self._run_python_in_env([COLLECT_EVAL_SCRIPT, '--root', EVAL_RESULTS_ROOT])
        if rc != 0:
            raise RuntimeError(f"collect_eval_results failed: rc={rc}, stderr={err.strip()}")
        try:
            logging.info(f"Collect results: {out.strip()}")
        except Exception:
            pass

    def _put_front(self, task: str) -> None:
        """Insert a job to the front of the queue (priority)."""
        # NOTE: access to underlying deque is not part of public API but works in practice
        with self.job_queue.mutex:
            self.job_queue.queue.appendleft(task)
            self.job_queue.unfinished_tasks += 1
            self.job_queue.not_empty.notify()

    @staticmethod
    def _is_eval_yaml(yaml_path: str) -> bool:
        try:
            base = os.path.basename(yaml_path)
            if base.startswith('EVAL_'):
                return True
            # heuristic: resides in an eval yaml dir
            parts = os.path.normpath(yaml_path).split(os.sep)
            return 'eval' in parts
        except Exception:
            return False

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
                            # Submit new job (.sh or YAML task)
                            task = data
                            accept = False
                            reason = ""
                            if task.endswith('.sh') and os.path.exists(task):
                                accept = True
                            elif task.endswith('.yaml') and os.path.exists(task):
                                accept = True
                            elif ':' in task:
                                try:
                                    prefix, path = task.split(':', 1)
                                    prefix = prefix.strip().lower()
                                    path = path.strip()
                                    if prefix in {'train', 'eval'} and os.path.exists(path) and path.endswith('.yaml'):
                                        accept = True
                                    else:
                                        reason = "Prefix not in {train,eval} or YAML path invalid."
                                except Exception as e:
                                    reason = f"Bad task format: {e}"
                            else:
                                reason = "Unknown task format. Submit a .sh, .yaml, or 'train:/path.yaml'/'eval:/path.yaml'."

                            if accept:
                                self.job_queue.put(task)
                                msg = f"✅ Job '{task}' submitted successfully. It's position #{self.job_queue.qsize()} in the queue."
                                logging.info(msg)
                                conn.sendall(msg.encode('utf-8'))
                            else:
                                msg = f"❌ Error: Task '{task}' not accepted. {reason}"
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
def submit_job(task_str):
    # Accept .sh, .yaml, or prefixed command like 'train:/path.yaml'
    if task_str.endswith('.sh'):
        abs_path = os.path.abspath(task_str)
        if not os.path.exists(abs_path):
            print(f"❌ Error: Script '{abs_path}' not found.")
            return
        payload = abs_path
    elif task_str.endswith('.yaml'):
        abs_path = os.path.abspath(task_str)
        if not os.path.exists(abs_path):
            print(f"❌ Error: YAML '{abs_path}' not found.")
            return
        payload = abs_path
    elif ':' in task_str:
        prefix, path = task_str.split(':', 1)
        path = path.strip()
        if prefix.strip().lower() not in {'train', 'eval'}:
            print("❌ Error: Prefix must be 'train' or 'eval'.")
            return
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path) or not abs_path.endswith('.yaml'):
            print(f"❌ Error: YAML '{abs_path}' not found or not a .yaml file.")
            return
        payload = f"{prefix}:{abs_path}"
    else:
        print("❌ Error: Unsupported task. Provide a .sh, .yaml, or 'train:/path.yaml'/'eval:/path.yaml'.")
        return
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            s.sendall(payload.encode('utf-8'))
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
    submit_parser = subparsers.add_parser('submit', help='Submit a shell script or a YAML task to the queue.')
    submit_parser.add_argument('task', type=str, help="Path to .sh/.yaml, or 'train:/abs/config.yaml'/'eval:/abs/config.yaml'.")
    subparsers.add_parser('status', help='Check the current status of the scheduler.')
    subparsers.add_parser('kill', help='Terminate the currently running job.')  # Added kill

    args = parser.parse_args()
    if args.command == 'start':
        scheduler = JobScheduler()
        scheduler.start()
    elif args.command == 'submit':
        submit_job(args.task)
    elif args.command == 'status':
        get_status()
    elif args.command == 'kill':
        kill_current()

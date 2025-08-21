## Scheduler usage

A lightweight queueing service to run one job at a time. Supports:

- Shell scripts: `.sh`
- YAML tasks for `llamafactory-cli`: `train:/abs/path.yaml`, `eval:/abs/path.yaml`, or just `/abs/path.yaml` (defaults to `train`).
  - You can also specify devices and evaluation behavior with `@` and `#no_eval`.

### Prerequisites

- Make sure `llamafactory-cli` is available in PATH. The simplest way is to start the scheduler from the same environment that has it installed. For example, if using conda env `lyh-lf`:

```bash
conda run -n lyh-lf python /data2/lyh/Custom-LLaMA-Factory/scheduler/scheduler.py start
```

### Start the service

```bash
python /data2/lyh/Custom-LLaMA-Factory/scheduler/scheduler.py start
```

You can also start it under your preferred environment (recommended):

```bash
conda run -n lyh-lf python /data2/lyh/Custom-LLaMA-Factory/scheduler/scheduler.py start
```

### Submit jobs

- **Shell script:**

```bash
python /data2/lyh/Custom-LLaMA-Factory/scheduler/scheduler.py submit /abs/path/to/run.sh
```

  - Only `.sh` files are accepted. Device and eval flags are not supported for `.sh` jobs.

- **YAML jobs:**

  - **Implicit train:**

```bash
python /data2/lyh/Custom-LLaMA-Factory/scheduler/scheduler.py submit /abs/path/to/config.yaml
```

  - **Specify devices:**

```bash
python /data2/lyh/Custom-LLaMA-Factory/scheduler/scheduler.py submit /abs/path/to/config.yaml@0,1,2,3
```

  - **Disable evaluation after training:**

```bash
python /data2/lyh/Custom-LLaMA-Factory/scheduler/scheduler.py submit /abs/path/to/config.yaml#no_eval
```

  - **Specify devices and disable eval:**

```bash
python /data2/lyh/Custom-LLaMA-Factory/scheduler/scheduler.py submit /abs/path/to/config.yaml@0,1,2,3#no_eval
```

  - **Explicit command (train or eval):**

```bash
python /data2/lyh/Custom-LLaMA-Factory/scheduler/scheduler.py submit "train:/abs/path/to/config.yaml"
python /data2/lyh/Custom-LLaMA-Factory/scheduler/scheduler.py submit "eval:/abs/path/to/config.yaml"
```

  - **With devices:**

```bash
python /data2/lyh/Custom-LLaMA-Factory/scheduler/scheduler.py submit "train:/abs/path/to/config.yaml@0,1,2,3"
python /data2/lyh/Custom-LLaMA-Factory/scheduler/scheduler.py submit "eval:/abs/path/to/config.yaml@0,1,2,3"
```

  - **With devices and disable eval:**

```bash
python /data2/lyh/Custom-LLaMA-Factory/scheduler/scheduler.py submit "train:/abs/path/to/config.yaml@0,1,2,3#no_eval"
```

#### Notes on job submission
- Only `.sh` and `.yaml` files are accepted. YAML jobs must exist and have the correct extension.
- `.sh` jobs do **not** support `@devices` or `#no_eval`.
- For YAML jobs, `@0,1,2,3` sets `ASCEND_RT_VISIBLE_DEVICES` for that job. `#no_eval` disables automatic evaluation after training.
- The client will print immediate acceptance or error. See logs for details.

### Check status

```bash
python /data2/lyh/Custom-LLaMA-Factory/scheduler/scheduler.py status
```

### Kill current job

```bash
python /data2/lyh/Custom-LLaMA-Factory/scheduler/scheduler.py kill
```

### Behavior for YAML tasks

- The scheduler reads `output_dir` from the YAML, creates it if needed, and copies the YAML into `output_dir` as `<name>_<timestamp>.yaml`.
- It runs `llamafactory-cli <command> <yaml>` and streams logs to terminal and to `output_dir/llamafactory_<command>_<timestamp>.log`.
- **After training:**
  - Runs a LoRA merge step (if applicable) and logs to `output_dir/merge_lora_<timestamp>.log`.
  - If evaluation is enabled (default), generates an eval YAML and enqueues it automatically.
- **After evaluation:**
  - Runs a results collector to refresh the summary CSV.

### Environment variables

The scheduler mirrors the environment used by the shell scripts. For YAML jobs (and `.sh` jobs for consistency), it sets defaults unless you override them in the service environment:

- `FORCE_TORCHRUN=1`
- `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` (or whatever value you have exported before starting the scheduler, or as set by `@devices` for YAML jobs)
- `SWANLAB_MODE=disabled` (can be overridden)
- `LLAMABOARD_WORKDIR=<output_dir>`

To customize, export before starting the scheduler:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export SWANLAB_MODE=disabled
conda run -n lyh-lf python /data2/lyh/Custom-LLaMA-Factory/scheduler/scheduler.py start
```

### Notes

- `.yaml` submissions default to `train` if not prefixed.
- Jobs are queued FIFO and executed one at a time.
- The client command prints immediate acceptance or error; detailed logs are in the scheduler terminal and in per-job log files (YAML tasks).
- After training, merge and (unless disabled) evaluation are handled automatically.
- After evaluation, results are collected automatically.



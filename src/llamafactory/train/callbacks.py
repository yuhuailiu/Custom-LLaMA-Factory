# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import json
import os
import random
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.distributed as dist
import transformers
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import PreTrainedModel, ProcessorMixin, TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, has_length
from transformers.utils import (
    SAFE_WEIGHTS_NAME,
    WEIGHTS_NAME,
    is_safetensors_available,
)
from typing_extensions import override

from ..extras import logging
from ..extras.constants import TRAINER_LOG, V_HEAD_SAFE_WEIGHTS_NAME, V_HEAD_WEIGHTS_NAME
from ..extras.misc import get_peak_memory, is_env_enabled, use_ray


if is_safetensors_available():
    from safetensors import safe_open
    from safetensors.torch import save_file


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, TrainerControl, TrainerState, TrainingArguments
    from trl import AutoModelForCausalLMWithValueHead

    from ..hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = logging.get_logger(__name__)


def fix_valuehead_checkpoint(
    model: "AutoModelForCausalLMWithValueHead", output_dir: str, safe_serialization: bool
) -> None:
    r"""Fix the valuehead checkpoint files.

    The model is already unwrapped.

    There are three cases:
    1. full tuning without ds_zero3: state_dict = {"model.layers.*": ..., "v_head.summary.*": ...}
    2. lora tuning without ds_zero3: state_dict = {"v_head.summary.*": ...}
    3. under deepspeed zero3: state_dict = {"pretrained_model.model.layers.*": ..., "v_head.summary.*": ...}

    We assume `stage3_gather_16bit_weights_on_model_save=true`.
    """
    if not isinstance(model.pretrained_model, (PreTrainedModel, PeftModel)):
        return

    if safe_serialization:
        path_to_checkpoint = os.path.join(output_dir, SAFE_WEIGHTS_NAME)
        with safe_open(path_to_checkpoint, framework="pt", device="cpu") as f:
            state_dict: dict[str, torch.Tensor] = {key: f.get_tensor(key) for key in f.keys()}
    else:
        path_to_checkpoint = os.path.join(output_dir, WEIGHTS_NAME)
        state_dict: dict[str, torch.Tensor] = torch.load(path_to_checkpoint, map_location="cpu", weights_only=True)

    os.remove(path_to_checkpoint)
    decoder_state_dict, v_head_state_dict = {}, {}
    for name, param in state_dict.items():
        if name.startswith("v_head."):
            v_head_state_dict[name] = param
        else:
            decoder_state_dict[name.replace("pretrained_model.", "", 1)] = param

    model.pretrained_model.save_pretrained(
        output_dir, state_dict=decoder_state_dict or None, safe_serialization=safe_serialization
    )

    if safe_serialization:
        save_file(v_head_state_dict, os.path.join(output_dir, V_HEAD_SAFE_WEIGHTS_NAME), metadata={"format": "pt"})
    else:
        torch.save(v_head_state_dict, os.path.join(output_dir, V_HEAD_WEIGHTS_NAME))

    logger.info_rank0(f"Value head model saved at: {output_dir}")


try:
    from deepspeed.accelerator import get_accelerator
except ImportError:
    def get_accelerator():
        return torch.cuda


try:
    import deepspeed
except ImportError:
    deepspeed = None


class FixValueHeadModelCallback(TrainerCallback):
    r"""A callback for fixing the checkpoint for valuehead models."""

    @override
    def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            fix_valuehead_checkpoint(
                model=kwargs.pop("model"), output_dir=output_dir, safe_serialization=args.save_safetensors
            )


class SaveProcessorCallback(TrainerCallback):
    r"""A callback for saving the processor."""

    def __init__(self, processor: "ProcessorMixin") -> None:
        self.processor = processor

    @override
    def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            self.processor.save_pretrained(output_dir)

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            self.processor.save_pretrained(args.output_dir)


class PissaConvertCallback(TrainerCallback):
    r"""A callback for converting the PiSSA adapter to a normal one."""

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            model = kwargs.pop("model")
            pissa_init_dir = os.path.join(args.output_dir, "pissa_init")
            logger.info_rank0(f"Initial PiSSA adapter will be saved at: {pissa_init_dir}.")
            if isinstance(model, PeftModel):
                init_lora_weights = getattr(model.peft_config["default"], "init_lora_weights")
                setattr(model.peft_config["default"], "init_lora_weights", True)
                model.save_pretrained(pissa_init_dir, safe_serialization=args.save_safetensors)
                setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            model = kwargs.pop("model")
            pissa_init_dir = os.path.join(args.output_dir, "pissa_init")
            pissa_backup_dir = os.path.join(args.output_dir, "pissa_backup")
            pissa_convert_dir = os.path.join(args.output_dir, "pissa_converted")
            logger.info_rank0(f"Converted PiSSA adapter will be saved at: {pissa_convert_dir}.")
            # 1. save a pissa backup with init_lora_weights: True
            # 2. save a converted lora with init_lora_weights: pissa
            # 3. load the pissa backup with init_lora_weights: True
            # 4. delete the initial adapter and change init_lora_weights to pissa
            if isinstance(model, PeftModel):
                init_lora_weights = getattr(model.peft_config["default"], "init_lora_weights")
                setattr(model.peft_config["default"], "init_lora_weights", True)
                model.save_pretrained(pissa_backup_dir, safe_serialization=args.save_safetensors)
                setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)
                model.save_pretrained(
                    pissa_convert_dir,
                    safe_serialization=args.save_safetensors,
                    path_initial_model_for_weight_conversion=pissa_init_dir,
                )
                model.load_adapter(pissa_backup_dir, "default", is_trainable=True)
                model.set_adapter("default")
                setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)


class SelektCallback(TrainerCallback):
    r"""A callback for applying the SeleKT selective knowledge transfer algorithm."""

    def __init__(
        self,
        base_model_state_dict: dict[str, torch.Tensor],
        alpha: float,
        selekt_steps: int,
        flush_steps: int = 1,
    ) -> None:
        self.base_model_state_dict = base_model_state_dict
        self.alpha = alpha
        self.selekt_steps = selekt_steps
        self.flush_steps = max(1, flush_steps)
        self.trainer = None

    def set_trainer(self, trainer) -> None:
        self.trainer = trainer

    @staticmethod
    def _sanitize_name(name: str) -> str:
        return name.replace("module.", "").replace("_orig_mod.", "")

    def _use_zero_stage3_rank0(self) -> bool:
        if self.trainer is None or deepspeed is None or not getattr(self.trainer, "is_deepspeed_enabled", False):
            return False
        engine = getattr(self.trainer, "model_wrapped", None)
        if engine is None:
            return False
        zero_stage = getattr(engine, "zero_optimization_stage", 0)
        if callable(zero_stage):
            zero_stage = zero_stage()
        return zero_stage == 3

    def _gather_context(self, param: torch.Tensor):
        if deepspeed is not None and getattr(self.trainer, "is_deepspeed_enabled", False):
            return deepspeed.zero.GatheredParameters(param, modifier_rank=0)
        return nullcontext()

    def _maybe_empty_cache(self) -> None:
        accelerator = get_accelerator()
        try:
            accelerator.empty_cache()
        except AttributeError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def on_step_end(self, args, state, control, **kwargs):
        if self.trainer is None or state.global_step == 0:
            return

        if self.flush_steps and state.global_step % self.flush_steps == 0:
            self._maybe_empty_cache()
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

        if self.selekt_steps <= 0 or state.global_step % self.selekt_steps != 0:
            return

        local_rank = int(os.getenv("LOCAL_RANK", os.getenv("RANK", "0")))
        print(f"SeleKT: global_step={state.global_step}, local_rank={local_rank}")
        self._apply_selekt_in_place(local_rank)

    def _apply_selekt_in_place(self, local_rank: int) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        model = getattr(self.trainer, "model_wrapped", getattr(self.trainer, "model", None))
        if model is None:
            return

        rank0_only = self._use_zero_stage3_rank0()
        with torch.no_grad():
            for name, param in tqdm(
                model.named_parameters(),
                desc="SeleKT",
                disable=(local_rank != 0),
                leave=False,
            ):
                if not param.requires_grad:
                    continue

                with self._gather_context(param):
                    # if rank0_only and local_rank != 0:
                    #     continue
                    if local_rank == 0:
                        clean_name = self._sanitize_name(name)
                        base_param = self.base_model_state_dict.get(clean_name)
                        if base_param is None:
                            continue

                        base_param = base_param.to(device=param.device, dtype=param.dtype, non_blocking=True)
                        delta = param.data - base_param

                        topk = int(self.alpha * delta.numel())
                        if topk <= 0:
                            param.data.copy_(base_param)
                            del base_param, delta
                            continue

                        topk = min(topk, delta.numel())
                        delta_abs = delta.abs()
                        delta_flat = delta_abs.view(-1)
                        _, indices = torch.topk(delta_flat, topk)

                        mask = torch.zeros_like(delta_flat)
                        mask[indices] = 1
                        mask = mask.view_as(delta)
                        delta.mul_(mask)

                        param.data.copy_(base_param + delta)

                        del base_param, delta, delta_abs, delta_flat, mask, indices

                if local_rank == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        for _ in range(3):
            gc.collect()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            if hasattr(torch.cuda, "reset_accumulated_memory_stats"):
                torch.cuda.reset_accumulated_memory_stats()

        self._maybe_empty_cache()


class LogCallback(TrainerCallback):
    r"""A callback for logging training and evaluation status."""

    def __init__(self) -> None:
        # Progress
        self.start_time = 0
        self.cur_steps = 0
        self.max_steps = 0
        self.elapsed_time = ""
        self.remaining_time = ""
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        # Status
        self.aborted = False
        self.do_train = False
        # Web UI
        self.webui_mode = is_env_enabled("LLAMABOARD_ENABLED")
        if self.webui_mode and not use_ray():
            signal.signal(signal.SIGABRT, self._set_abort)
            self.logger_handler = logging.LoggerHandler(os.getenv("LLAMABOARD_WORKDIR"))
            logging.add_handler(self.logger_handler)
            transformers.logging.add_handler(self.logger_handler)

    def _set_abort(self, signum, frame) -> None:
        self.aborted = True

    def _reset(self, max_steps: int = 0) -> None:
        self.start_time = time.time()
        self.cur_steps = 0
        self.max_steps = max_steps
        self.elapsed_time = ""
        self.remaining_time = ""

    def _timing(self, cur_steps: int) -> None:
        cur_time = time.time()
        elapsed_time = cur_time - self.start_time
        avg_time_per_step = elapsed_time / cur_steps if cur_steps != 0 else 0
        remaining_time = (self.max_steps - cur_steps) * avg_time_per_step
        self.cur_steps = cur_steps
        self.elapsed_time = str(timedelta(seconds=int(elapsed_time)))
        self.remaining_time = str(timedelta(seconds=int(remaining_time)))

    def _write_log(self, output_dir: str, logs: dict[str, Any]) -> None:
        with open(os.path.join(output_dir, TRAINER_LOG), "a", encoding="utf-8") as f:
            f.write(json.dumps(logs) + "\n")

    def _create_thread_pool(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.thread_pool = ThreadPoolExecutor(max_workers=1)

    def _close_thread_pool(self) -> None:
        if self.thread_pool is not None:
            self.thread_pool.shutdown(wait=True)
            self.thread_pool = None

    @override
    def on_init_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if (
            args.should_save
            and os.path.exists(os.path.join(args.output_dir, TRAINER_LOG))
            and args.overwrite_output_dir
        ):
            logger.warning_rank0_once("Previous trainer log in this folder will be deleted.")
            os.remove(os.path.join(args.output_dir, TRAINER_LOG))

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            self.do_train = True
            self._reset(max_steps=state.max_steps)
            self._create_thread_pool(output_dir=args.output_dir)

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        self._close_thread_pool()

    @override
    def on_substep_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if self.aborted:
            control.should_epoch_stop = True
            control.should_training_stop = True

    @override
    def on_step_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if self.aborted:
            control.should_epoch_stop = True
            control.should_training_stop = True

    @override
    def on_evaluate(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not self.do_train:
            self._close_thread_pool()

    @override
    def on_predict(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not self.do_train:
            self._close_thread_pool()

    @override
    def on_log(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not args.should_save:
            return

        self._timing(cur_steps=state.global_step)
        logs = dict(
            current_steps=self.cur_steps,
            total_steps=self.max_steps,
            loss=state.log_history[-1].get("loss"),
            eval_loss=state.log_history[-1].get("eval_loss"),
            predict_loss=state.log_history[-1].get("predict_loss"),
            reward=state.log_history[-1].get("reward"),
            accuracy=state.log_history[-1].get("rewards/accuracies"),
            lr=state.log_history[-1].get("learning_rate"),
            epoch=state.log_history[-1].get("epoch"),
            percentage=round(self.cur_steps / self.max_steps * 100, 2) if self.max_steps != 0 else 100,
            elapsed_time=self.elapsed_time,
            remaining_time=self.remaining_time,
        )
        if state.num_input_tokens_seen:
            logs["throughput"] = round(state.num_input_tokens_seen / (time.time() - self.start_time), 2)
            logs["total_tokens"] = state.num_input_tokens_seen

        if is_env_enabled("RECORD_VRAM"):
            vram_allocated, vram_reserved = get_peak_memory()
            logs["vram_allocated"] = round(vram_allocated / (1024**3), 2)
            logs["vram_reserved"] = round(vram_reserved / (1024**3), 2)

        logs = {k: v for k, v in logs.items() if v is not None}
        if self.webui_mode and all(key in logs for key in ("loss", "lr", "epoch")):
            log_str = f"'loss': {logs['loss']:.4f}, 'learning_rate': {logs['lr']:2.4e}, 'epoch': {logs['epoch']:.2f}"
            for extra_key in ("reward", "accuracy", "throughput"):
                if logs.get(extra_key):
                    log_str += f", '{extra_key}': {logs[extra_key]:.2f}"

            logger.info_rank0("{" + log_str + "}")

        if self.thread_pool is not None:
            self.thread_pool.submit(self._write_log, args.output_dir, logs)

    @override
    def on_prediction_step(
        self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs
    ):
        if self.do_train:
            return

        if self.aborted:
            sys.exit(0)

        if not args.should_save:
            return

        eval_dataloader = kwargs.pop("eval_dataloader", None)
        if has_length(eval_dataloader):
            if self.max_steps == 0:
                self._reset(max_steps=len(eval_dataloader))
                self._create_thread_pool(output_dir=args.output_dir)

            self._timing(cur_steps=self.cur_steps + 1)
            if self.cur_steps % 5 == 0 and self.thread_pool is not None:
                logs = dict(
                    current_steps=self.cur_steps,
                    total_steps=self.max_steps,
                    percentage=round(self.cur_steps / self.max_steps * 100, 2) if self.max_steps != 0 else 100,
                    elapsed_time=self.elapsed_time,
                    remaining_time=self.remaining_time,
                )
                self.thread_pool.submit(self._write_log, args.output_dir, logs)


class ReporterCallback(TrainerCallback):
    r"""A callback for reporting training status to external logger."""

    def __init__(
        self,
        model_args: "ModelArguments",
        data_args: "DataArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
    ) -> None:
        self.model_args = model_args
        self.data_args = data_args
        self.finetuning_args = finetuning_args
        self.generating_args = generating_args
        os.environ["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT", "llamafactory")

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not state.is_world_process_zero:
            return

        if "wandb" in args.report_to:
            import wandb

            wandb.config.update(
                {
                    "model_args": self.model_args.to_dict(),
                    "data_args": self.data_args.to_dict(),
                    "finetuning_args": self.finetuning_args.to_dict(),
                    "generating_args": self.generating_args.to_dict(),
                }
            )

        if self.finetuning_args.use_swanlab:
            import swanlab  # type: ignore

            swanlab.config.update(
                {
                    "model_args": self.model_args.to_dict(),
                    "data_args": self.data_args.to_dict(),
                    "finetuning_args": self.finetuning_args.to_dict(),
                    "generating_args": self.generating_args.to_dict(),
                }
            )


class EvalAdoptationCallback(TrainerCallback):
    r"""
    A callback for evaluating model generation quality at specific training steps.
    
    This callback:
    1. Triggers evaluation at user-specified steps (after start_step, every eval_steps)
    2. Reads evaluation data from a JSONL file
    3. Generates model responses for each prompt (no chat template, greedy decoding)
    4. Computes metrics using adoptation_eval.process_item
    5. Saves results to output directory with step-wise metrics
    
    Usage:
        callback = EvalAdoptationCallback(
            tokenizer=tokenizer,
            jsonl_path="/path/to/eval.jsonl",
            eval_steps=500,
            start_step=100,
            output_dir="/path/to/output",
        )
    """

    def __init__(
        self,
        tokenizer: "PreTrainedTokenizer",
        jsonl_path: str,
        eval_steps: int = 500,
        start_step: int = 0,
        max_new_tokens: int = 512,
        output_dir: Optional[str] = None,
        prompt_key: str = "prompt",
        ground_truth_key: str = "ground_truth",
        eval_fim_mode: bool = True,
        eval_at_end: bool = True,
        max_input_length: int = 4096,
        eval_batch_size: int = 4,
        save_best_model: bool = True,
    ) -> None:
        """
        Initialize the EvalAdoptationCallback.
        
        Args:
            tokenizer: The tokenizer for encoding/decoding.
            jsonl_path: Path to the JSONL file containing evaluation data.
            eval_steps: Evaluate every N steps after start_step.
            start_step: Start evaluation after this step.
            max_new_tokens: Maximum new tokens to generate.
            output_dir: Directory to save evaluation results.
            prompt_key: Key name for prompt in the JSONL file.
            ground_truth_key: Key name for ground truth in the JSONL file.
            eval_fim_mode: Whether to use FIM mode for evaluation.
            eval_at_end: Whether to evaluate at the end of training.
        """
        self.tokenizer = tokenizer
        self.jsonl_path = jsonl_path
        self.eval_steps = eval_steps
        self.start_step = start_step
        self.max_new_tokens = max_new_tokens
        self.output_dir = output_dir
        self.prompt_key = prompt_key
        self.ground_truth_key = ground_truth_key
        self.eval_fim_mode = eval_fim_mode
        self.eval_at_end = eval_at_end
        self.trainer = None
        self._evaluated_steps: set[int] = set()
        self._eval_history: list[dict] = []  # 记录每次评估的结果
        self.max_input_length = max_input_length
        self.eval_batch_size = eval_batch_size

        # === 新增：最佳模型跟踪 ===
        self.save_best_model = save_best_model
        # 以 avg_代码采纳率 作为最佳模型指标
        self.best_metric_name = "avg_代码采纳率"
        self.best_metric_value: float = float("-inf")
        self.best_model_step: Optional[int] = None
        
        # 加载评估数据
        self._eval_data: list[dict] = []
        self._encoded_eval_inputs: list[dict] = []
        self._load_eval_data()

        self.eos_token_ids: list[int] = []
        base_eos = getattr(self.tokenizer, "eos_token_id", None)
        if base_eos is not None:
            self.eos_token_ids.append(base_eos)
        for special_token in ["<|im_end|>", "<|eot_id|>", "</s>", "<|end|>", "<|endoftext|>"]:
            try:
                tid = self.tokenizer.convert_tokens_to_ids(special_token)
                if (
                    tid is not None
                    and tid != self.tokenizer.unk_token_id
                    and tid not in self.eos_token_ids
                ):
                    self.eos_token_ids.append(tid)
            except Exception:
                pass
        
        # 导入评测模块
        from ..extras.adoptation_eval import process_item
        self._process_item = process_item
        
        logger.info_rank0(f"EvalAdoptationCallback initialized with {len(self._eval_data)} samples from {jsonl_path}")
    
    def _maybe_save_best_model(
        self,
        model: torch.nn.Module,
        avg_metrics: dict,
        global_step: int,
    ) -> None:
        if not self.save_best_model:
            return
        if not self.output_dir:
            return

        metric_value = avg_metrics.get(self.best_metric_name)
        if metric_value is None:
            # 当前评估没算出这个指标，直接跳过
            return

        # 只有采纳率严格高于历史最佳才保存
        if metric_value <= self.best_metric_value:
            return

        self.best_metric_value = metric_value
        self.best_model_step = global_step

        best_root = os.path.join(self.output_dir, "best")
        best_model_dir = os.path.join(best_root, "full_model")
        best_adapter_dir = os.path.join(best_root, "adapter")
        os.makedirs(best_root, exist_ok=True)

        unwrapped = self._unwrap_model(model)

        # 保存完整模型（包含 tokenizer）
        try:
            unwrapped.save_pretrained(best_model_dir)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(best_model_dir)
            logger.info_rank0(
                f"New best full model saved at step {global_step} "
                f"({self.best_metric_name}={metric_value:.4f}) -> {best_model_dir}"
            )
        except Exception as e:
            logger.warning_rank0(f"Failed to save best full model: {e}")

        # 如果是 LoRA / PeftModel，单独保存 adapter
        try:
            from peft import PeftModel  # type: ignore
            if isinstance(unwrapped, PeftModel):
                os.makedirs(best_adapter_dir, exist_ok=True)
                unwrapped.save_pretrained(best_adapter_dir)
                logger.info_rank0(
                    f"New best adapter saved at step {global_step} "
                    f"({self.best_metric_name}={metric_value:.4f}) -> {best_adapter_dir}"
                )
        except Exception as e:
            # peft 不在环境里或者不是 PeftModel，就忽略
            logger.warning_rank0(f"Failed to save best adapter (ignored): {e}")

        # 记录一个 info 文件，方便之后加载
        info = {
            "best_step": self.best_model_step,
            "best_metric_name": self.best_metric_name,
            "best_metric_value": self.best_metric_value,
        }
        info_path = os.path.join(best_root, "best_model_info.json")
        try:
            with open(info_path, "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning_rank0(f"Failed to write best_model_info.json: {e}")

    def _load_eval_data(self) -> None:
        """Load evaluation data from JSONL file."""
        if not self.jsonl_path or not os.path.exists(self.jsonl_path):
            logger.warning_rank0(f"EvalAdoptationCallback: JSONL file not found: {self.jsonl_path}")
            return
        
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning_rank0(f"Failed to parse JSON line: {e}")
                    continue

                prompt = item.get(self.prompt_key, "")
                if not prompt:
                    # 没有 prompt 的样本直接丢掉，和之前“后面跳过”等价
                    continue

                # 保存原始样本
                self._eval_data.append(item)

                # NEW: 预先 tokenize 到 CPU 上
                enc = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_input_length,
                )
                self._encoded_eval_inputs.append(enc)

    def set_trainer(self, trainer) -> None:
        """Set the trainer reference."""
        self.trainer = trainer
        if self.output_dir is None and hasattr(trainer, 'args'):
            self.output_dir = trainer.args.output_dir

    def _should_evaluate(self, global_step: int) -> bool:
        """Check if we should evaluate at this step."""
        if global_step in self._evaluated_steps:
            return False
        
        if global_step < self.start_step:
            return False
        
        # 从 start_step 开始，每 eval_steps 步评估一次
        steps_since_start = global_step - self.start_step
        return steps_since_start >= 0 and steps_since_start % self.eval_steps == 0

    def _get_device(self) -> torch.device:
        """Get the device for generation."""
        if self.trainer is not None and hasattr(self.trainer, 'accelerator'):
            return self.trainer.accelerator.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _is_main_process(self) -> bool:
        """Check if current process is main process."""
        local_rank = int(os.getenv("LOCAL_RANK", os.getenv("RANK", "0")))
        return local_rank == 0

    def _unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Unwrap model from DeepSpeed/FSDP/DataParallel wrappers.
        Works with both full fine-tuning and LoRA.
        """
        # 尝试从 trainer 获取 unwrapped model
        if self.trainer is not None:
            # accelerate 的 unwrap_model
            if hasattr(self.trainer, 'accelerator') and hasattr(self.trainer.accelerator, 'unwrap_model'):
                return self.trainer.accelerator.unwrap_model(model)
        
        # DeepSpeed wrapper
        if hasattr(model, 'module'):
            return model.module
        
        # DataParallel / DistributedDataParallel
        if hasattr(model, '_orig_mod'):
            return model._orig_mod
        
        return model
        
    def _generate_batch(self, model: torch.nn.Module, batch_encodings: list[dict]) -> list[str]:
        """
        Generate model responses for a batch of pre-tokenized inputs.
        No chat template, greedy decoding (do_sample=False).
        使用左 padding 以兼容 decoder-only 架构的正确生成。
        """
        device = self._get_device()
        unwrapped_model = self._unwrap_model(model)

        # 单条 input_ids / attention_mask 列表
        input_ids_list = [e["input_ids"].squeeze(0) for e in batch_encodings]
        attn_list = [e["attention_mask"].squeeze(0) for e in batch_encodings]

        # 统一长度，用左 padding
        max_len = max(t.size(0) for t in input_ids_list)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        batch_size = len(input_ids_list)
        batch_input_ids = torch.full(
            (batch_size, max_len),
            fill_value=pad_id,
            dtype=torch.long,
        )
        batch_attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=torch.long,
        )

        for i, (ids, mask) in enumerate(zip(input_ids_list, attn_list)):
            L = ids.size(0)
            # 左 padding：把真实 token 放到右边
            batch_input_ids[i, max_len - L: max_len] = ids
            batch_attention_mask[i, max_len - L: max_len] = mask

        batch_input_ids = batch_input_ids.to(device)
        batch_attention_mask = batch_attention_mask.to(device)

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            pad_token_id=pad_id,
            do_sample=False,
        )
        if self.eos_token_ids:
            gen_kwargs["eos_token_id"] = self.eos_token_ids

        outputs = unwrapped_model.generate(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            **gen_kwargs,
        )

        # 对每个样本，去掉整个输入序列（含 PAD + prompt），只保留新生成部分
        full_input_len = batch_input_ids.size(1)  # == max_len
        generated_texts: list[str] = []
        for i in range(batch_size):
            out = outputs[i]
            gen_tokens = out[full_input_len:]
            text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
            generated_texts.append(text)

        return generated_texts

    def _run_evaluation(self, model: torch.nn.Module, global_step: int) -> None:
        """Run the actual evaluation. Only runs on main process to avoid distributed issues."""
        if not self._eval_data:
            logger.warning_rank0("EvalAdoptationCallback: No evaluation data loaded, skipping.")
            return
        
        # 标记该步骤已评估
        self._evaluated_steps.add(global_step)
        is_main = self._is_main_process()
        
        # 只在主进程上执行评估
        if not is_main:
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            return
        
        logger.info_rank0(f"EvalAdoptationCallback: Running evaluation at step {global_step}")
        
        was_training = model.training

        # 在 eval 阶段临时打开 use_cache，加速自回归解码
        orig_use_cache = None
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            orig_use_cache = model.config.use_cache
            model.config.use_cache = True

        model.eval()
        results = []

        num_samples = len(self._eval_data)
        batch_size = max(1, self.eval_batch_size)

        # eval 用 inference_mode 包裹整个过程
        with torch.inference_mode():
            iterator = tqdm(
                range(0, num_samples, batch_size),
                desc=f"Evaluating (step {global_step})",
                leave=False,
            )

            for start in iterator:
                end = min(start + batch_size, num_samples)
                batch_items = self._eval_data[start:end]
                batch_encodings = self._encoded_eval_inputs[start:end]

                # 批量 generate
                batch_responses = self._generate_batch(model, batch_encodings)

                for item, generated_response in zip(batch_items, batch_responses):
                    try:
                        ground_truth = item.get(self.ground_truth_key, "")

                        result_item = dict(item)
                        result_item["generated_response"] = generated_response

                        # 将原始 item 的所有字段传入，同时添加/覆盖生成结果和 ground_truth
                        eval_input = dict(item)
                        eval_input["First_Chunk"] = generated_response
                        eval_input["real_ground_truth"] = ground_truth
                        
                        eval_result = self._process_item(
                            eval_input,
                            generated_key="First_Chunk",
                            ground_truth_key="real_ground_truth",
                            eval_fim=self.eval_fim_mode,
                        )

                        result_item.update(eval_result)
                        results.append(result_item)
                    except Exception as e:
                        logger.warning_rank0(f"Failed to process sample: {e}")
                        continue
        
        # 恢复模型状态
        if was_training:
            model.train()

        # 恢复 use_cache 原值
        if hasattr(model, "config") and orig_use_cache is not None:
            model.config.use_cache = orig_use_cache
        
        # 保存结果
        if results:
            self._save_and_log_results(results, global_step, model)
        
        # 同步所有进程
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def _save_and_log_results(
        self,
        results: list[dict],
        global_step: int,
        model: torch.nn.Module,
    ) -> None:
        """Save evaluation results and log metrics."""
        if not results:
            return
        
        # 计算平均指标
        metrics = {
            '相似度': [],
            '代码召回率': [],
            '代码采纳率': [],
            'F1_score': [],
            '首行命中率': [],
            '前2行命中率': [],
            '前3行命中率': [],
            '前4行命中率': [],
            '前5行命中率': [],
        }
        
        for r in results:
            for key in metrics:
                if key in r:
                    metrics[key].append(r[key])
        
        avg_metrics = {}
        for key, values in metrics.items():
            if values:
                avg_metrics[f"avg_{key}"] = round(sum(values) / len(values), 4)
        
        avg_metrics['step'] = global_step
        avg_metrics['num_samples'] = len(results)
        
        # 记录到历史
        self._eval_history.append(avg_metrics)
        
        # 打印关键指标
        logger.info_rank0(
            f"Step {global_step} Evaluation Results: "
            f"采纳率={avg_metrics.get('avg_代码采纳率', 0):.2f}%, "
            f"召回率={avg_metrics.get('avg_代码召回率', 0):.2f}%, "
            f"相似度={avg_metrics.get('avg_相似度', 0):.2f}%, "
            f"首行命中率={avg_metrics.get('avg_首行命中率', 0):.2f}%"
        )
        # save best model
        self._maybe_save_best_model(model, avg_metrics, global_step)
        
        # 保存结果
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            
            # 保存详细结果
            detail_path = os.path.join(self.output_dir, f"adoptation_eval_step_{global_step}.jsonl")
            with open(detail_path, 'w', encoding='utf-8') as f:
                for item in results:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            # 保存/更新汇总结果
            summary_path = os.path.join(self.output_dir, "adoptation_eval_summary.jsonl")
            with open(summary_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(avg_metrics, ensure_ascii=False) + '\n')
            
            logger.info_rank0(f"Evaluation results saved to {detail_path}")

    @override
    def on_step_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        """Called at the end of each training step."""
        model = kwargs.get("model")
        if model is None:
            return
        
        global_step = state.global_step
        
        if self._should_evaluate(global_step):
            self._run_evaluation(model, global_step)

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        """Called at the end of training."""
        if not self.eval_at_end:
            return
        
        model = kwargs.get("model")
        if model is None:
            return
        
        global_step = state.global_step
        
        # 在训练结束时进行最终评估（如果尚未评估过该步骤）
        if global_step not in self._evaluated_steps:
            logger.info_rank0(f"EvalAdoptationCallback: Running final evaluation at step {global_step}")
            self._run_evaluation(model, global_step)
        
        # 保存最终汇总
        if self._is_main_process() and self._eval_history and self.output_dir:
            final_summary_path = os.path.join(self.output_dir, "adoptation_eval_final_summary.json")
            with open(final_summary_path, 'w', encoding='utf-8') as f:
                json.dump(self._eval_history, f, ensure_ascii=False, indent=2)
            logger.info_rank0(f"Final evaluation summary saved to {final_summary_path}")


# def _generate_response_single(self, model: torch.nn.Module, prompt: str) -> str:
#         """
#         Generate model response for a given prompt.
#         No chat template, greedy decoding (do_sample=False).
#         Works with both full fine-tuning and LoRA (PeftModel).
#         """
#         device = self._get_device()
        
#         # Unwrap model to get the actual model for generation
#         unwrapped_model = self._unwrap_model(model)
        
#         # Tokenize directly without chat template
#         inputs = self.tokenizer(
#             prompt,
#             return_tensors="pt",
#             truncation=True,
#             max_length=16384,  # 较大的max_length以适应长prompt
#         )
#         inputs = {k: v.to(device) for k, v in inputs.items()}
        
#         # 确定 eos_token_ids
#         eos_token_ids = [self.tokenizer.eos_token_id]
#         for special_token in ["<|im_end|>", "<|eot_id|>", "</s>", "<|end|>", "<|endoftext|>"]:
#             try:
#                 token_id = self.tokenizer.convert_tokens_to_ids(special_token)
#                 if token_id is not None and token_id != self.tokenizer.unk_token_id and token_id not in eos_token_ids:
#                     eos_token_ids.append(token_id)
#             except Exception:
#                 pass
        
#         # Generate with greedy decoding
#         # 对于 PeftModel (LoRA), generate() 会自动应用 LoRA 适配器
#         with torch.no_grad():
#             outputs = unwrapped_model.generate(
#                 **inputs,
#                 max_new_tokens=self.max_new_tokens,
#                 pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
#                 eos_token_id=eos_token_ids,
#                 do_sample=False,  # 贪婪解码
#             )
        
#         # Decode only the generated part
#         generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
#         generated_response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
#         return generated_response

#     def _run_evaluation_single(self, model: torch.nn.Module, global_step: int) -> None:
#         """Run the actual evaluation. Only runs on main process to avoid distributed issues."""
#         if not self._eval_data:
#             logger.warning_rank0("EvalAdoptationCallback: No evaluation data loaded, skipping.")
#             return
        
#         # 标记该步骤已评估
#         self._evaluated_steps.add(global_step)
        
#         is_main = self._is_main_process()
        
#         # 只在主进程上执行评估，避免分布式训练（特别是 DeepSpeed ZeRO-3）的问题
#         if not is_main:
#             # 等待主进程完成评估
#             if dist.is_available() and dist.is_initialized():
#                 dist.barrier()
#             return
        
#         logger.info_rank0(f"EvalAdoptationCallback: Running evaluation at step {global_step}")
        
#         was_training = model.training
#         model.eval()
        
#         results = []
        
#         with torch.no_grad():
#             iterator = tqdm(
#                 self._eval_data,
#                 desc=f"Evaluating (step {global_step})",
#                 leave=False,
#             )
            
#             for item in iterator:
#                 try:
#                     prompt = item.get(self.prompt_key, "")
#                     ground_truth = item.get(self.ground_truth_key, "")
                    
#                     if not prompt:
#                         continue
                    
#                     # 生成响应（对于 LoRA，会自动应用适配器）
#                     generated_response = self._generate_response(model, prompt)
                    
#                     # 复制原始数据并添加生成结果
#                     result_item = dict(item)
#                     result_item['generated_response'] = generated_response
                    
#                     # 调用 process_item 进行评估
#                     eval_result = self._process_item(
#                         {
#                             'First_Chunk': generated_response,
#                             'real_ground_truth': ground_truth,
#                         },
#                         generated_key='First_Chunk',
#                         ground_truth_key='real_ground_truth',
#                         eval_fim=self.eval_fim_mode,
#                     )
                    
#                     # 合并评估结果
#                     result_item.update(eval_result)
#                     results.append(result_item)
                    
#                 except Exception as e:
#                     logger.warning_rank0(f"Failed to process sample: {e}")
#                     continue
        
#         # 恢复模型状态
#         if was_training:
#             model.train()
        
#         # 保存结果（此时已经在主进程中）
#         if results:
#             self._save_and_log_results(results, global_step)
        
#         # 同步所有进程
#         if dist.is_available() and dist.is_initialized():
#             dist.barrier()
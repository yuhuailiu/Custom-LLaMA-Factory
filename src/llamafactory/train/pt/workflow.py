# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
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

import math
import os
import gc
from typing import TYPE_CHECKING, Optional

import torch
from transformers import DataCollatorForLanguageModeling, AutoModelForCausalLM

from ...data import get_dataset, get_template_and_fix_tokenizer
from ...extras.logging import get_logger
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..callbacks import SelektCallback
from ..trainer_utils import create_modelcard_and_push
from .trainer import CustomTrainer

try:
    import deepspeed
except ImportError:
    deepspeed = None


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments


logger = get_logger(__name__)


def run_pt(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="pt", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)

    runtime_callbacks: list["TrainerCallback"] = list(callbacks) if callbacks is not None else []
    selekt_callback: Optional[SelektCallback] = None
    if finetuning_args.use_selekt:
        if not training_args.do_train:
            raise ValueError("SeleKT requires training to be enabled.")

        logger.info_rank0(
            f"Preparing SeleKT base model snapshot (alpha={finetuning_args.selekt_alpha}, "
            f"steps={finetuning_args.selekt_steps})."
        )

        # NOTE:
        # - For SeleKT we only need a frozen snapshot of the *initial* base model weights.
        # - When DeepSpeed ZeRO-3 is enabled, loading this snapshot as a trainable model can
        #   interact badly with DeepSpeed's patched `state_dict` logic and yield empty tensors
        #   (shape == 0) in the resulting state dict.
        # - To avoid this, we deliberately load the base model in inference mode
        #   (`is_trainable=False`), which bypasses those training-time patches while still
        #   giving us the correct full-precision weights.
        base_model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, torch_dtype=torch.bfloat16)
        base_model.cpu()
        if deepspeed is not None:
            from deepspeed import zero
            with zero.GatheredParameters(list(base_model.parameters()), modifier_rank=0):
                if os.getenv("LOCAL_RANK", os.getenv("RANK", "0")) == "0":
                    raw_sd = base_model.state_dict()
                    base_state_dict = {
                        k: v.detach().clone().cpu()  # 关键是 clone() + cpu()
                        for k, v in raw_sd.items()
                    }
                else:
                    base_state_dict = {}
            # 这里可以再用 dist.broadcast / accelerate 广播给其它 rank
        else:
            raw_sd = base_model.state_dict()
            base_state_dict = {k: v.detach().clone().cpu() for k, v in raw_sd.items()}
        del base_model
        for _ in range(3):
            gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        selekt_callback = SelektCallback(
            base_model_state_dict=base_state_dict,
            alpha=finetuning_args.selekt_alpha,
            selekt_steps=finetuning_args.selekt_steps,
        )
        runtime_callbacks.append(selekt_callback)
        logger.info_rank0("SeleKT callback enabled for full-parameter PT.")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Initialize our Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=runtime_callbacks,
        **dataset_module,
        **tokenizer_module,
    )

    if selekt_callback is not None:
        selekt_callback.set_trainer(trainer)

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            keys = ["loss"]
            if isinstance(dataset_module.get("eval_dataset"), dict):
                keys += [f"eval_{key}_loss" for key in dataset_module["eval_dataset"].keys()]
            else:
                keys += ["eval_loss"]

            plot_loss(training_args.output_dir, keys=keys)

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")

        if isinstance(dataset_module.get("eval_dataset"), dict):
            for key in dataset_module["eval_dataset"].keys():
                try:
                    perplexity = math.exp(metrics[f"eval_{key}_loss"])
                except OverflowError:
                    perplexity = float("inf")

                metrics[f"eval_{key}_perplexity"] = perplexity
        else:
            try:
                perplexity = math.exp(metrics["eval_loss"])
            except OverflowError:
                perplexity = float("inf")

            metrics["eval_perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)

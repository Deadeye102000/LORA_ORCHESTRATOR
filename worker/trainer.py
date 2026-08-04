import os
import sys
import json
import time
import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone LoRA/QLoRA Trainer Subprocess"
    )
    parser.add_argument("--job_id", type=int, required=True, help="Job ID")
    parser.add_argument(
        "--base_model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Base causal LM model (e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0)",
    )
    parser.add_argument(
        "--dataset_url",
        type=str,
        required=True,
        help="HuggingFace dataset name or local JSON path",
    )
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument("--epochs", type=float, default=3.0, help="Number of epochs")
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size per device"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to train on: cpu or cuda",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
#  MOCK TRAINING  (CPU fallback — no GPU / no model download required)
# ─────────────────────────────────────────────────────────────────────────────


def run_mock_training(job_id, epochs, steps_per_epoch=5):
    """Simulates a training run without any real GPU/model."""
    print(f"Initializing MOCK training run for Job {job_id}...", flush=True)
    print("Loading mock base model configuration...", flush=True)
    print(f"Starting MOCK training for Job {job_id}...", file=sys.stderr)
    total_steps = int(epochs * steps_per_epoch)
    loss = 2.5
    for step in range(1, total_steps + 1):
        time.sleep(0.5)  # Simulate compute work
        loss -= 0.15 * (loss / 2.0)  # Gradual decay
        epoch = round(step / steps_per_epoch, 2)
        metrics = {
            "step": step,
            "loss": round(loss, 4),
            "epoch": epoch,
        }
        print(json.dumps(metrics), flush=True)

    weights_dir = f"./weights/job_{job_id}"
    os.makedirs(weights_dir, exist_ok=True)
    with open(os.path.join(weights_dir, "adapter_config.json"), "w") as f:
        json.dump(
            {"base_model_name_or_path": "mock", "peft_type": "LORA"},
            f,
        )
    with open(os.path.join(weights_dir, "adapter_model.safetensors"), "w") as f:
        f.write("MOCK_WEIGHTS_DATA")

    print(
        f"Mock training completed. Weights saved to {weights_dir}",
        file=sys.stderr,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  REAL QLORA TRAINING  (requires CUDA GPU)
#  Uses:  AutoModelForCausalLM + BitsAndBytesConfig (4-bit QLoRA)
#         peft LoraConfig targeting q_proj / v_proj
#         trl SFTTrainer for instruction-tuning loop
# ─────────────────────────────────────────────────────────────────────────────


def run_real_training(
    job_id,
    base_model,
    dataset_url,
    lora_r,
    lora_alpha,
    learning_rate,
    epochs,
    batch_size,
):
    print(
        f"[Job {job_id}] Starting REAL QLoRA training on model: {base_model}",
        file=sys.stderr,
    )

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainerCallback,
    )
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from trl import SFTConfig, SFTTrainer
    from datasets import load_dataset, Dataset

    # ── 1. 4-bit QLoRA quantization config ────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # NormalFloat4 — best quality
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,  # Double quantization saves VRAM
    )

    # ── 2. Load tokenizer ─────────────────────────────────────────────────────
    model_name = base_model or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"[Job {job_id}] Loading tokenizer: {model_name}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Required for causal LM batching

    # ── 3. Load base model with 4-bit quantization ────────────────────────────
    print(f"[Job {job_id}] Loading 4-bit quantised model...", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",  # Distributes layers across available GPUs
        trust_remote_code=True,
    )
    # Prepare for k-bit training: casts norms to fp32, enables gradient checkpointing
    model = prepare_model_for_kbit_training(model)

    # ── 4. LoRA configuration (Causal LM targets: q_proj, v_proj) ─────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        # Standard Llama-family attention projection targets for LoRA
        target_modules=["q_proj", "v_proj"],
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    print(
        f"[Job {job_id}] Trainable params: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)",
        file=sys.stderr,
    )

    # ── 5. Load dataset ───────────────────────────────────────────────────────
    #  Supports:
    #   - HuggingFace dataset name  (e.g. "timdettmers/openassistant-guanaco")
    #   - Local JSON file path      (e.g. "/data/train.json")
    print(f"[Job {job_id}] Loading dataset: {dataset_url}", file=sys.stderr)
    try:
        if dataset_url.endswith(".json") and os.path.exists(dataset_url):
            raw_dataset = load_dataset("json", data_files=dataset_url, split="train")
        else:
            raw_dataset = load_dataset(dataset_url, split="train")
    except Exception as e:
        print(
            f"[Job {job_id}] Dataset load failed ({e}), using fallback.",
            file=sys.stderr,
        )
        # Fallback: tiny in-memory instruction dataset
        raw_dataset = Dataset.from_dict(
            {
                "text": [
                    "### Instruction:\nWhat is LoRA?\n### Response:\n"
                    "LoRA is a parameter-efficient fine-tuning method.",
                    "### Instruction:\nExplain gradient descent.\n"
                    "### Response:\nGradient descent minimises loss by "
                    "iteratively adjusting weights.",
                    "### Instruction:\nWhat is a transformer?\n"
                    "### Response:\nA transformer is a deep learning "
                    "architecture based on self-attention mechanisms.",
                ]
                * 30
            }
        )

    # ── 6. Metric streaming callback ──────────────────────────────────────────
    class LogStdoutCallback(TrainerCallback):
        """Streams step metrics as JSON to stdout for the daemon to capture."""

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                loss = logs.get("loss")
                if loss is not None:
                    metrics = {
                        "step": state.global_step,
                        "loss": round(loss, 4),
                        "epoch": round(state.epoch, 2),
                    }
                    print(json.dumps(metrics), flush=True)

    # ── 7. SFT training config ────────────────────────────────────────────────
    weights_dir = f"./weights/job_{job_id}"
    sft_config = SFTConfig(
        output_dir=weights_dir,
        num_train_epochs=int(epochs),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        bf16=True,  # Use bfloat16 on Ampere+ GPUs
        optim="paged_adamw_8bit",  # 8-bit AdamW — compatible with QLoRA
        logging_steps=1,
        save_strategy="no",  # Weights saved manually after training
        report_to="none",
        dataset_text_field="text",
        max_seq_length=512,
    )

    # ── 8. Run SFTTrainer ─────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=raw_dataset,
        tokenizer=tokenizer,
        callbacks=[LogStdoutCallback()],
    )

    print(f"[Job {job_id}] Starting SFTTrainer training loop...", file=sys.stderr)
    trainer.train()

    # ── 9. Save LoRA adapter weights ──────────────────────────────────────────
    model.save_pretrained(weights_dir)
    tokenizer.save_pretrained(weights_dir)
    print(
        f"[Job {job_id}] Training complete. Adapter saved to {weights_dir}",
        file=sys.stderr,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    print(
        f"Trainer subprocess started with arguments: {vars(args)}",
        flush=True,
    )

    use_cuda = args.device == "cuda" and torch.cuda.is_available()

    if not use_cuda:
        print(
            "[Trainer] CUDA unavailable — running mock training.",
            file=sys.stderr,
        )
        run_mock_training(args.job_id, args.epochs)
    else:
        try:
            run_real_training(
                job_id=args.job_id,
                base_model=args.base_model,
                dataset_url=args.dataset_url,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                learning_rate=args.learning_rate,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
        except Exception as e:
            print(
                f"[Trainer] Real training failed: {e}. "
                "Falling back to mock training.",
                file=sys.stderr,
            )
            run_mock_training(args.job_id, args.epochs)


if __name__ == "__main__":
    main()

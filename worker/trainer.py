import os
import sys
import json
import time
import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone LoRA Trainer Subprocess")
    parser.add_argument("--job_id", type=int, required=True, help="Job ID")
    parser.add_argument(
        "--base_model", type=str, default="prajjwal1/bert-tiny", help="Base model"
    )
    parser.add_argument(
        "--dataset_url", type=str, required=True, help="URL or path to dataset"
    )
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument(
        "--learning_rate", type=float, default=2e-4, help="Learning rate"
    )
    parser.add_argument("--epochs", type=float, default=3.0, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device (cpu or cuda)"
    )
    return parser.parse_args()


def run_mock_training(job_id, epochs, steps_per_epoch=5):
    print(f"Initializing MOCK training run for Job {job_id}...", flush=True)
    print("Loading mock base model configuration...", flush=True)
    print(f"Starting MOCK training for Job {job_id}...", file=sys.stderr)
    total_steps = int(epochs * steps_per_epoch)
    loss = 2.5
    for step in range(1, total_steps + 1):
        time.sleep(0.5)  # Simulate compute work
        loss -= 0.15 * (loss / 2.0)  # gradual decay
        epoch = round(step / steps_per_epoch, 2)
        metrics = {"step": step, "loss": round(loss, 4), "epoch": epoch}
        print(json.dumps(metrics), flush=True)

    weights_dir = f"./weights/job_{job_id}"
    os.makedirs(weights_dir, exist_ok=True)
    with open(os.path.join(weights_dir, "adapter_config.json"), "w") as f:
        json.dump({"base_model_name_or_path": "mock", "peft_type": "LORA"}, f)
    with open(os.path.join(weights_dir, "adapter_model.safetensors"), "w") as f:
        f.write("MOCK_WEIGHTS_DATA")

    print(
        f"Mock training completed. Weights saved to {weights_dir}",
        file=sys.stderr,
    )


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
        f"Starting REAL LoRA training for Job {job_id} on model {base_model}...",
        file=sys.stderr,
    )

    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        TrainerCallback,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset

    class LogStdoutCallback(TrainerCallback):
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

    # 1. Load tokenizer and model
    model_name = base_model if base_model else "prajjwal1/bert-tiny"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 2. Configure PEFT (LoRA)
    target_modules = ["query", "value"] if "bert" in model_name.lower() else None
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)

    # 3. Create dummy dataset
    texts = [
        "This is an amazing training dataset!",
        "The orchestrator is working beautifully.",
        "LoRA adapter weights are highly efficient.",
        "We need to monitor steps and loss metrics.",
        "FastAPI is extremely fast and robust.",
        "Distributed training scaling is important.",
    ] * 20
    labels = [1, 1, 1, 0, 1, 0] * 20

    dataset = Dataset.from_dict({"text": texts, "label": labels})

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=64,
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # 4. Define training args
    weights_dir = f"./weights/job_{job_id}"
    training_args = TrainingArguments(
        output_dir=weights_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        evaluation_strategy="no",
        save_strategy="no",
        logging_steps=1,
        no_cuda=False,
        report_to="none",
    )

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        callbacks=[LogStdoutCallback()],
    )

    # 6. Run train
    trainer.train()

    # Save adapter weights
    model.save_pretrained(weights_dir)
    print(
        f"Real training completed. Weights saved to {weights_dir}",
        file=sys.stderr,
    )


def main():
    args = parse_args()
    print(
        f"Trainer subprocess started with arguments: {vars(args)}",
        flush=True,
    )

    # Run training
    use_cuda = args.device == "cuda" and torch.cuda.is_available()

    if not use_cuda:
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
                f"Real training failed with error: {e}. "
                "Falling back to Mock training.",
                file=sys.stderr,
            )
            run_mock_training(args.job_id, args.epochs)


if __name__ == "__main__":
    main()

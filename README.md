# Multi-Cost Knowledge Distillation for Language Models

---

## Environment Setup

- **Python Version:** Requires Python 3.10 or 3.11.
- **Dependencies:** Install all required packages using:
  ```bash
  pip install -r requirements.txt
  ```

---

## Training and Fine-tuning

### Model Training

- **GPT2-base**
  ```bash
  bash scripts/gpt2/multicost_gpt2_base.sh
  ```
- **TinyLLaMA-1.1B**
  ```bash
  bash scripts/tinyllama/multicost_tinyllama.sh
  ```

### Supervised Fine-tuning (SFT) for Teacher Models

- **Qwen1.5-1.8B** (full fine-tuning)
  ```bash
  bash scripts/gpt2/sft_teacher_qwen.sh
  ```
- **LLaMA2-7B** (LoRA-based)
  ```bash
  bash scripts/tinyllama/sft_teacher_llama2.sh
  ```
- **Mistral-7B** (LoRA-based)
  ```bash
  bash scripts/tinyllama/sft_teacher_mistral.sh
  ```

> Pretrained SFT models for Qwen1.5-1.8B, Mistral-7B (LoRA), and Qwen2.5-7B-Instruct are available here: [Google Drive](https://drive.google.com/drive/folders/11Eba3lgnWZGjFW2EPUQVC1nLv6-LRI-9?usp=sharing)

### SFT for Student Models

- **GPT2-base** (full fine-tuning)
  ```bash
  bash scripts/gpt2/sft_gpt2_base.sh
  ```
- **TinyLLaMA-1.1B** (LoRA-based)
  ```bash
  bash scripts/tinyllama/sft_tinyllama.sh
  ```

**Note:**  
If you encounter errors when loading TinyLLaMA checkpoints due to version mismatches of the `transformers` library (TinyLLaMA recommends v4.31), refer to [this issue](https://github.com/songmzhang/DSKD/issues/8) for a solution.

---

## Knowledge Distillation Baselines

### Dual-Space KD with CMA
- **GPT2-base**
  ```bash
  bash scripts/gpt2/dskd_cma_gpt2_base.sh
  ```
- **TinyLLaMA-1.1B**
  ```bash
  bash scripts/tinyllama/dskd_cma_tinyllama.sh
  ```

### Logits Alignment with Minimum Edit Distance
- **GPT2-base**
  ```bash
  bash scripts/gpt2/minedit_gpt2_base.sh
  ```
- **TinyLLaMA-1.1B**
  ```bash
  bash scripts/tinyllama/minedit_tinyllama.sh
  ```

### Universal Logit Distillation
- **GPT2-base**
  ```bash
  bash scripts/gpt2/uld_gpt2_base.sh
  ```
- **TinyLLaMA-1.1B**
  ```bash
  bash scripts/tinyllama/uld_tinyllama.sh
  ```

### Standard Dual-Space KD
- **GPT2-base**
  ```bash
  bash scripts/gpt2/dskd_gpt2_base.sh
  ```
- **TinyLLaMA-1.1B**
  ```bash
  bash scripts/tinyllama/dskd_tinyllama.sh
  ```

### Vanilla Knowledge Distillation
- **GPT2-base**
  ```bash
  bash scripts/gpt2/vanilla_kd_gpt2_base.sh
  ```
- **TinyLLaMA-1.1B**
  ```bash
  bash scripts/tinyllama/vanilla_kd_tinyllama.sh
  ```

> You can modify the KD objective (e.g., KL, Reverse KL, JS, etc.) in these scripts using the `KD_OBJ` variable.

---

## Output Structure

- **Full Fine-tuning:**  
  Output is stored in `./outputs/<model>/<model-size>/sft/criterion=.../`
  ```
  ./outputs/gpt2/gpt2-base/sft/criterion=.../
  ├── epochA_step.../           # Model checkpoint directory
  │   ├── config.json
  │   ├── pytorch_model.bin
  │   ├── tokenizer.json
  │   └── ...
  ├── args.json                 # Training arguments
  └── train.log                 # Training log
  ```
- **LoRA Fine-tuning:**  
  Output is stored in `./outputs/<model>/<model-size>/sft/criterion=.../`
  ```
  ./outputs/tinyllama/tinyllama-1.1b-3T/sft/criterion=.../
  ├── epochA_step.../           # LoRA checkpoint directory
  │   ├── adapter_config.json
  │   ├── adapter_model.bin
  │   ├── tokenizer.json
  │   └── ...
  ├── args.json
  └── train.log
  ```

---

## Evaluation

### Full Fine-tuning Checkpoints

```bash
bash scripts/eval/run_eval.sh <CKPT_PATH> <EVAL_BATCH_SIZE>
```
- `CKPT_PATH`: Absolute path to the model checkpoint directory (e.g., `/home/xxx/outputs/gpt2/gpt2-base/sft/.../epochA_step...`).

### LoRA Fine-tuning Checkpoints

```bash
bash scripts/eval/run_eval_lora.sh <LORA_ADAPTER_PATH> <EVAL_BATCH_SIZE>
```
- `LORA_ADAPTER_PATH`: Absolute path to the LoRA adapter directory (e.g., `/home/xxx/outputs/tinyllama/tinyllama-1.1b-3T/sft/.../epochA_step...`).
- You may need to set `MODEL_PATH` in `run_eval_lora.sh` to the appropriate base model.

---

## Data

- Processed datasets for training and evaluation are available here: [Google Drive Link](https://drive.google.com/drive/folders/1ZUsNVgWevACV9D-AHVNi9C7PX_2itzb8?usp=sharing)

---

## Model Weights

Download pre-trained or base model weights into `model_hub/<model>/<variant>/`.  
Links to official Hugging Face model repositories:

- [GPT2-120M](https://huggingface.co/openai-community/gpt2)
- [GPT2-1.5B (Dolly)](https://github.com/microsoft/LMOps/blob/main/minillm/README.md#31-resources)
- [Qwen1.5-1.8B](https://huggingface.co/Qwen/Qwen1.5-1.8B)
- [TinyLLaMA-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T)
- [Llama2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)

---

## Notes

- All scripts are provided in the `scripts/` directory for straightforward training and evaluation.
- For troubleshooting or environment issues (such as package versions), refer to relevant GitHub issues or open a new one in this repo.
- If you use or adapt this codebase, please cite the original authors and papers for the respective methods and models.

---
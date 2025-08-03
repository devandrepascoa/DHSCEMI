import shutil
from enum import verify
from pathlib import Path
from huggingface_hub import hf_hub_download

sample_config = {
    "models": [
        {
            "repo_id": "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
            "filename": "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
            "local_name": "01-Deepseek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
        },
        {
            "repo_id": "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
            "filename": "DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf",
            "local_name": "02-Deepseek-R1-Distill-Qwen-1.5B-Q8_0.gguf",
        },
        {
            "repo_id": "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
            "filename": "DeepSeek-R1-Distill-Qwen-1.5B-BF16.gguf",
            "local_name": "03-Deepseek-R1-Distill-Qwen-1.5B-BF16.gguf",
        },
        {
            "repo_id": "unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF",
            "filename": "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
            "local_name": "04-Deepseek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
        },
        {
            "repo_id": "unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF",
            "filename": "DeepSeek-R1-Distill-Qwen-7B-Q8_0.gguf",
            "local_name": "05-Deepseek-R1-Distill-Qwen-7B-Q8_0.gguf",
        },
        {
            "repo_id": "unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF",
            "filename": "DeepSeek-R1-Distill-Qwen-7B-F16.gguf",
            "local_name": "06-Deepseek-R1-Distill-Qwen-7B-F16.gguf",
        },
        {
            "repo_id": "unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF",
            "filename": "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
            "local_name": "07-Deepseek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
        },
        {
            "repo_id": "unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF",
            "filename": "DeepSeek-R1-Distill-Qwen-14B-Q8_0.gguf",
            "local_name": "08-Deepseek-R1-Distill-Qwen-14B-Q8_0.gguf",
        },
        {
            "repo_id": "unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF",
            "filename": "DeepSeek-R1-Distill-Qwen-14B-F16.gguf",
            "local_name": "09-Deepseek-R1-Distill-Qwen-14B-F16.gguf",
        },
        {
            "repo_id": "unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF",
            "filename": "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf",
            "local_name": "10-Deepseek-R1-Distill-Qwen-32B-Q4_K_M.gguf",
        }
    ]
}

models_dir = Path("./models")
models_dir.mkdir(parents=True, exist_ok=True)


for model in sample_config["models"]:
    downloaded_path = hf_hub_download(
        repo_id=model["repo_id"],
        filename=model["filename"],
    )
    shutil.copy2(downloaded_path, models_dir / model["local_name"])

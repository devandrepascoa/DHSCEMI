import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

sample_config = {
    "models": [
        {
            "repo_id": "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
            "filename": "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
            "local_name": "01-DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf",
        },
        {
            "repo_id": "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
            "filename": "DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf",
            "local_name": "02-DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf",
        },
        {
            "repo_id": "unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
            "filename": "DeepSeek-R1-Distill-Qwen-1.5B-BF16.gguf",
            "local_name": "03-DeepSeek-R1-Distill-Qwen-1.5B-BF16.gguf",
        },
        {
            "repo_id": "unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF",
            "filename": "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
            "local_name": "04-DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
        },
        {
            "repo_id": "unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF",
            "filename": "DeepSeek-R1-Distill-Qwen-7B-Q8_0.gguf",
            "local_name": "05-DeepSeek-R1-Distill-Qwen-7B-Q8_0.gguf",
        },
        {
            "repo_id": "unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF",
            "filename": "DeepSeek-R1-Distill-Qwen-7B-F16.gguf",
            "local_name": "06-DeepSeek-R1-Distill-Qwen-7B-F16.gguf",
        },
        {
            "repo_id": "unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF",
            "filename": "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
            "local_name": "07-DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
        },
        {
            "repo_id": "unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF",
            "filename": "DeepSeek-R1-Distill-Qwen-14B-Q8_0.gguf",
            "local_name": "08-DeepSeek-R1-Distill-Qwen-14B-Q8_0.gguf",
        },
        {
            "repo_id": "unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF",
            "filename": "DeepSeek-R1-Distill-Qwen-14B-F16.gguf",
            "local_name": "09-DeepSeek-R1-Distill-Qwen-14B-F16.gguf",
        },
        {
            "repo_id": "unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF",
            "filename": "DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf",
            "local_name": "10-DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf",
        },
        # {
        #     "repo_id": "unsloth/Qwen3-1.7B-GGUF",
        #     "filename": "Qwen3-1.7B-Q4_K_M.gguf",
        #     "local_name": "11-Qwen3-1.7B-Q4_K_M.gguf",
        # },
        # {
        #     "repo_id": "unsloth/Qwen3-1.7B-GGUF",
        #     "filename": "Qwen3-1.7B-Q8_0.gguf",
        #     "local_name": "12-Qwen3-1.7B-Q8_0.gguf",
        # },
        # {
        #     "repo_id": "unsloth/Qwen3-1.7B-GGUF",
        #     "filename": "Qwen3-1.7B-F16.gguf",
        #     "local_name": "13-Qwen3-1.7B-BF16.gguf"
        # },
        # {
        #     "repo_id": "unsloth/Qwen3-8B-GGUF",
        #     "filename": "Qwen3-8B-Q4_K_M.gguf",
        #     "local_name": "14-Qwen3-8B-Q4_K_M.gguf",
        # },
        # {
        #     "repo_id": "unsloth/Qwen3-8B-GGUF",
        #     "filename": "Qwen3-8B-Q8_0.gguf",
        #     "local_name": "15-Qwen3-8B-Q8_0.gguf",
        # },
        # {
        #     "repo_id": "unsloth/Qwen3-8B-GGUF",
        #     "filename": "Qwen3-8B-F16.gguf",
        #     "local_name": "16-Qwen3-8B-BF16.gguf"
        # },
    ]
}

models_dir = Path("./models")
models_dir.mkdir(parents=True, exist_ok=True)

print(f"Starting download of {len(sample_config['models'])} models...")
print(f"Models will be saved to: {models_dir.absolute()}")
print()

for model in sample_config["models"]:
    target_path = models_dir / model["local_name"]

    print(target_path)

    if target_path.exists():
        print(f"✓ {model['local_name']} already exists, skipping download")
        continue

    print(f"Downloading {model['local_name']}...")
    downloaded_path = hf_hub_download(
        repo_id=model["repo_id"],
        filename=model["filename"],
    )
    shutil.copy2(downloaded_path, target_path)
    print(f"✓ Downloaded {model['local_name']}")

print("Download process completed!")

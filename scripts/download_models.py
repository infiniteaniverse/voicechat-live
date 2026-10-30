"""
One-shot model downloader. Run this once per Part before implementing that Part.
After running for all Parts, zero network calls happen at pipeline runtime.

Usage:
    python scripts/download_models.py --part 1    # STT + VAD (run before Part 1)
    python scripts/download_models.py --part 3    # Embeddings (run before Part 3)
    python scripts/download_models.py --part 4    # Kokoro TTS (run before Part 4)
    python scripts/download_models.py --all       # Everything at once
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def download_whisper(model_size: str = "small") -> None:
    print(f"\n[STT] Downloading faster-whisper '{model_size}'...")
    from faster_whisper import WhisperModel
    # Instantiating with device="cpu" for the download — avoids CUDA init overhead
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    del model
    print(f"[STT] Done. Cached at: ~/.cache/huggingface/hub/")


def download_silero_vad() -> None:
    print("\n[VAD] Downloading Silero VAD...")
    import torch
    torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        verbose=True,
        trust_repo=True,
    )
    print("[VAD] Done. Cached at: ~/.cache/torch/hub/")


def download_embeddings() -> None:
    print("\n[EMB] Downloading all-MiniLM-L6-v2 (used by long-term memory in Part 3)...")
    from sentence_transformers import SentenceTransformer
    SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("[EMB] Done.")


def download_kokoro() -> None:
    print("\n[TTS] Downloading Kokoro-82M (used by TTS in Part 4)...")
    # Kokoro downloads its weights on first use via the kokoro package
    import kokoro
    # Trigger download by initializing a pipeline
    from kokoro import KPipeline
    pipeline = KPipeline(lang_code="a")  # 'a' = American English
    del pipeline
    print("[TTS] Done.")


def pull_ollama_model(model_tag: str = "qwen2.5:3b") -> None:
    print(f"\n[LLM] Pulling Ollama model '{model_tag}'...")
    print("      (Make sure Ollama is running: start it with 'ollama serve')")
    result = subprocess.run(["ollama", "pull", model_tag], check=False)
    if result.returncode != 0:
        print("[LLM] ✗ Failed. Is Ollama installed and running?")
        print("      Install: https://ollama.com")
        print(f"      Then run: ollama pull {model_tag}")
    else:
        print(f"[LLM] Done. Model '{model_tag}' is ready.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download VoiceChat model weights.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--part", type=int, choices=[1, 2, 3, 4],
                       help="Download models for a specific Part")
    group.add_argument("--all", action="store_true",
                       help="Download everything at once")
    args = parser.parse_args()

    import yaml
    with open(ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)

    model_size = config["stt"]["model_size"]
    llm_model = config["llm"]["model"]

    if args.all or args.part == 1:
        download_whisper(model_size)
        download_silero_vad()

    if args.all or args.part == 2:
        pull_ollama_model(llm_model)

    if args.all or args.part == 3:
        download_embeddings()

    if args.all or args.part == 4:
        download_kokoro()

    print("\n✓ All requested models downloaded. Zero network calls needed at runtime.")


if __name__ == "__main__":
    main()

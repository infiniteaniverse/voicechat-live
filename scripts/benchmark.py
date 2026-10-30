"""
Per-component latency benchmark. Grows with each Part.

Part 1: STT benchmark
Part 4: TTS benchmark
Part 5: Guardrails benchmark
Part 9: VAD benchmark + full-pipeline benchmark (completing the suite)

Usage:
    python scripts/benchmark.py --component stt
    python scripts/benchmark.py --component all

Note (Windows console): this file prints Unicode box-drawing characters. If you
hit a UnicodeEncodeError on cp1252, run with PYTHONIOENCODING=utf-8 set rather
than editing this file — see docs/tradeoffs.md.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def load_config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def _report(name: str, latencies: list[float], budget_ms: float, extra: str = "") -> None:
    latencies = sorted(latencies)
    median = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[0]
    p_min, p_max = latencies[0], latencies[-1]
    status = "PASS" if median < budget_ms else "FAIL - over budget"

    print(f"\n  {'-'*45}")
    print(f"  Median  : {median:.0f}ms")
    print(f"  P95     : {p95:.0f}ms")
    print(f"  Min/Max : {p_min:.0f}ms / {p_max:.0f}ms")
    print(f"  Budget  : < {budget_ms:.0f}ms")
    print(f"  Status  : {status}")
    if extra:
        print(f"  {extra}")
    print(f"  {'-'*45}\n")
    print(f"  -> Paste these numbers into docs/latency_report.md")


# ── VAD Benchmark (Part 9) ───────────────────────────────────────────────────

def benchmark_vad(config: dict, runs: int = 20) -> None:
    from audio.vad import SileroVAD

    print(f"\n{'-'*55}")
    print(f"  VAD Benchmark")
    print(f"  Model   : Silero VAD (torch.hub)")
    print(f"  Runs    : {runs}")
    print(f"{'-'*55}")

    vad = SileroVAD(config)
    window = np.zeros(512, dtype=np.float32)  # one Silero window (silence — measures pure inference cost)

    print("  Warming up...")
    vad.process_chunk(window)
    vad.reset()

    latencies: list[float] = []
    for i in range(runs):
        t0 = time.perf_counter()
        vad.process_chunk(window)
        ms = (time.perf_counter() - t0) * 1000
        latencies.append(ms)
        vad.reset()  # isolate single-window inference cost, not state-machine bookkeeping
        print(f"  Run {i+1:02d}: {ms:.1f}ms")

    # CLAUDE.md's original per-turn budget line: "VAD: ~50ms"
    _report("vad", latencies, budget_ms=50)


# ── STT Benchmark (Part 1) ────────────────────────────────────────────────────

def benchmark_stt(config: dict, runs: int = 20) -> None:
    from audio.stt import WhisperSTT

    model_size = config["stt"]["model_size"]
    device = config["stt"]["device"]
    compute_type = config["stt"]["compute_type"]

    print(f"\n{'-'*55}")
    print(f"  STT Benchmark")
    print(f"  Model   : faster-whisper {model_size}")
    print(f"  Device  : {device} ({compute_type})")
    print(f"  Runs    : {runs}")
    print(f"{'-'*55}")

    stt = WhisperSTT(config)

    # 5 seconds of low-amplitude white noise (Whisper handles empty audio gracefully)
    sample_rate = config["audio"]["sample_rate"]
    test_audio = (np.random.randn(5 * sample_rate) * 0.05).astype(np.float32)

    # Warm-up run (first inference is always slower due to CUDA/kernel init)
    print("  Warming up...")
    stt.transcribe(test_audio)

    latencies: list[float] = []
    for i in range(runs):
        t0 = time.perf_counter()
        stt.transcribe(test_audio)
        ms = (time.perf_counter() - t0) * 1000
        latencies.append(ms)
        print(f"  Run {i+1:02d}: {ms:.0f}ms")

    # Budget: STT should consume < 400ms of the 2s total budget
    _report("stt", latencies, budget_ms=400)


# ── TTS Benchmark (Part 4) ────────────────────────────────────────────────────

def benchmark_tts(config: dict, runs: int = 20) -> None:
    from audio.tts import create_tts_engine

    print(f"\n{'-'*55}")
    print(f"  TTS Benchmark")
    print(f"  Engine  : {config['tts']['engine']}")
    print(f"  Runs    : {runs}")
    print(f"{'-'*55}")

    tts = create_tts_engine(config)
    sentence = "Machine learning engineers build systems that learn from data."

    print("  Warming up...")
    tts.synthesize(sentence)

    latencies: list[float] = []
    for i in range(runs):
        t0 = time.perf_counter()
        tts.synthesize(sentence)
        ms = (time.perf_counter() - t0) * 1000
        latencies.append(ms)
        print(f"  Run {i+1:02d}: {ms:.0f}ms")

    # Matches tests/test_tts.py's own per-sentence synthesis budget
    _report("tts", latencies, budget_ms=300, extra="Requires CUDA torch — see docs/tradeoffs.md if this fails badly.")


# ── Guardrails Benchmark (Part 5) ─────────────────────────────────────────────

def benchmark_guardrails(config: dict, runs: int = 10) -> None:
    from guardrails.manager import GuardrailsManager

    print(f"\n{'-'*55}")
    print(f"  Guardrails Benchmark (requires Ollama running)")
    print(f"  Runs    : {runs}")
    print(f"{'-'*55}")

    manager = GuardrailsManager(config)
    input_text = "What's a good way to relax after work?"
    output_text = "Reading a book or going for a walk are both great ways to unwind."

    async def _run() -> tuple[list[float], list[float]]:
        print("  Warming up...")
        await manager.check_input(input_text)
        await manager.check_output(output_text)

        input_latencies: list[float] = []
        for i in range(runs):
            t0 = time.perf_counter()
            await manager.check_input(input_text)
            ms = (time.perf_counter() - t0) * 1000
            input_latencies.append(ms)
            print(f"  [input]  Run {i+1:02d}: {ms:.0f}ms")

        output_latencies: list[float] = []
        for i in range(runs):
            t0 = time.perf_counter()
            await manager.check_output(output_text)
            ms = (time.perf_counter() - t0) * 1000
            output_latencies.append(ms)
            print(f"  [output] Run {i+1:02d}: {ms:.0f}ms")

        return input_latencies, output_latencies

    try:
        input_latencies, output_latencies = asyncio.run(_run())
    except Exception as exc:
        print(f"\n  Guardrails benchmark failed — is `ollama serve` running? ({exc})")
        return

    print("\n  Input guardrail:")
    _report("guardrails_input", input_latencies, budget_ms=500)
    print("  Output guardrail:")
    _report("guardrails_output", output_latencies, budget_ms=500)


# ── Full Pipeline Benchmark (Part 9) ──────────────────────────────────────────

def benchmark_full_pipeline(config: dict, runs: int = 5) -> None:
    """Real turn latency: synthesizes a genuine spoken question with the
    project's own Kokoro TTS, feeds it through the real VAD/STT/guardrails/
    LLM/TTS via ConversationPipeline.run_turn() (same technique used for
    manual verification throughout Parts 7-8), and reports total per-turn
    latency against the 2s budget. Requires Ollama running.
    """
    import scipy.signal

    from pipeline.pipeline import ConversationPipeline

    print(f"\n{'-'*55}")
    print(f"  Full Pipeline Benchmark (requires Ollama running)")
    print(f"  Runs    : {runs}")
    print(f"{'-'*55}")

    bench_config = copy.deepcopy(config)
    tmp_dir = Path(tempfile.mkdtemp())
    bench_config["memory"]["long_term"]["store_path"] = str(tmp_dir / "faiss_index")
    bench_config["memory"]["long_term"]["facts_db_path"] = str(tmp_dir / "facts.sqlite")

    mic_rate = bench_config["audio"]["sample_rate"]
    chunk_size = bench_config["audio"]["chunk_size"]
    trailing_silence_s = 0.9

    def _make_feeder(pipeline: "ConversationPipeline", utterance: np.ndarray):
        def _feed() -> np.ndarray:
            pipeline.vad.reset()
            chunk_duration_s = chunk_size / mic_rate
            for i in range(0, len(utterance), chunk_size):
                chunk = utterance[i : i + chunk_size]
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                time.sleep(chunk_duration_s)
                result = pipeline.vad.process_chunk(chunk)
                if result is not None:
                    return result
            raise RuntimeError("VAD never fired on the synthetic benchmark utterance")

        return _feed

    async def _run() -> list[float]:
        print("  Building pipeline (isolated temp memory store) + pre-warming...")
        pipeline = ConversationPipeline(bench_config)
        await pipeline.prewarm()

        question = "What is the capital of France?"
        audio_24k = pipeline.tts.synthesize(question)
        audio_16k = scipy.signal.resample_poly(audio_24k, up=2, down=3).astype(np.float32)
        utterance = np.concatenate(
            [audio_16k, np.zeros(int(trailing_silence_s * mic_rate), dtype=np.float32)]
        )

        # time_to_first_audio = cumulative elapsed through the tts_first_chunk
        # mark — the actual "2s budget" metric this project measures
        # throughout. Summing stage_durations() instead would double-count
        # tts_complete's real-time playback duration as if it were
        # processing time, which it isn't.
        time_to_first_audio: list[float] = []
        for i in range(runs):
            pipeline._listen_for_utterance = _make_feeder(pipeline, utterance)
            await pipeline.run_turn()
            durations = pipeline.monitor.stage_durations()
            first_audio_ms = pipeline.monitor.elapsed_ms("tts_first_chunk")
            time_to_first_audio.append(first_audio_ms)
            stage_str = " | ".join(f"{k}: {v:.0f}ms" for k, v in durations.items())
            print(f"  Run {i+1:02d}: time_to_first_audio={first_audio_ms:.0f}ms  [{stage_str}]")

        await pipeline.close()
        return time_to_first_audio

    try:
        time_to_first_audio = asyncio.run(_run())
    except Exception as exc:
        print(f"\n  Full pipeline benchmark failed — is `ollama serve` running? ({exc})")
        return

    # Processing latency (VAD-fired -> first audio out) budget, per docs/latency_report.md
    _report(
        "full_pipeline",
        time_to_first_audio,
        budget_ms=2000,
        extra="This is time to first audio out — excludes the ~700ms mandatory VAD "
              "silence-detection wait and all playback time after the first sentence "
              "starts. See docs/latency_report.md for that breakdown.",
    )


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="VoiceChat component latency benchmarks.")
    parser.add_argument(
        "--component",
        choices=["vad", "stt", "tts", "guardrails", "pipeline", "all"],
        default="stt",
        help="Which component to benchmark",
    )
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()

    config = load_config()

    if args.component in ("vad", "all"):
        benchmark_vad(config, runs=args.runs)

    if args.component in ("stt", "all"):
        benchmark_stt(config, runs=args.runs)

    if args.component in ("tts", "all"):
        benchmark_tts(config, runs=args.runs)

    if args.component in ("guardrails", "all"):
        benchmark_guardrails(config, runs=min(args.runs, 10))

    if args.component in ("pipeline", "all"):
        benchmark_full_pipeline(config, runs=min(args.runs, 5))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Score module-level attribution risks for a QA dataset."""

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from knowledge_neurons.knowledge_neurons import KnowledgeNeurons


def read_jsonl(path: Path) -> List[Tuple[str, str]]:
    """Load question/answer pairs from a JSONL file."""

    qa_pairs: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc

            question = record.get("question")
            answer = record.get("answer")
            if not isinstance(question, str) or not isinstance(answer, str):
                raise ValueError(
                    f"Expected string 'question' and 'answer' fields on line {line_no}"
                )
            qa_pairs.append((question, answer))
    if not qa_pairs:
        raise ValueError(f"No valid question/answer pairs found in {path}")
    return qa_pairs


def tensor_to_list(tensor: torch.Tensor) -> List[List[float]]:
    """Convert a tensor to a nested Python list of floats."""

    if tensor.numel() == 0:
        return tensor.tolist()
    return tensor.detach().cpu().tolist()


def save_results(
    output_dir: Path,
    qa_pairs: Iterable[Tuple[str, str]],
    scores: dict,
) -> None:
    """Persist per-module attribution statistics to disk."""

    output_dir.mkdir(parents=True, exist_ok=True)

    questions = [q for q, _ in qa_pairs]
    answers = [a for _, a in qa_pairs]

    serializable = {
        "questions": questions,
        "answers": answers,
        "modules": {},
    }

    for module_name, per_prompt in scores["per_prompt"].items():
        module_entry = {
            "per_prompt": tensor_to_list(per_prompt),
            "mean": tensor_to_list(scores["mean"][module_name]),
            "max": tensor_to_list(scores["max"][module_name]),
        }
        serializable["modules"][module_name] = module_entry

    output_path = output_dir / "module_risk_scores.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path, help="Path to the autoregressive model")
    parser.add_argument("dataset", type=Path, help="Path to the safety risk JSONL dataset")
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where attribution results will be stored",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device identifier (defaults to CUDA if available)",
    )
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for IG sampling")
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of interpolation steps for integrated gradients",
    )
    parser.add_argument(
        "--max-answer-tokens",
        type=int,
        default=20,
        help="Maximum number of answer tokens to attribute",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars during scoring",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    qa_pairs = read_jsonl(args.dataset)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    kn = KnowledgeNeurons(
        model,
        tokenizer,
        device=device,
        max_target_tokens=args.max_answer_tokens,
    )

    scores = kn.score_module_risks(
        qa_pairs,
        steps=args.steps,
        batch_size=args.batch_size,
        max_answer_tokens=args.max_answer_tokens,
        pbar=not args.no_progress,
    )

    save_results(args.output_dir, qa_pairs, scores)


if __name__ == "__main__":
    main()

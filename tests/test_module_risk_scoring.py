import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parents[1]))

from knowledge_neurons import KnowledgeNeurons


class DummyBatch(dict):
    def to(self, device):
        for key, value in self.items():
            self[key] = value.to(device)
        return self


class DummyTokenizer:
    def __init__(self):
        self.vocab = {
            "<pad>": 0,
            "question": 1,
            "answer": 2,
            "token": 3,
            "long": 4,
            "response": 5,
            "short": 6,
            "reply": 7,
        }
        self._inv_vocab = {v: k for k, v in self.vocab.items()}
        self.pad_token_id = self.vocab["<pad>"]

    def encode(self, text: str, add_special_tokens: bool = False):
        return [self.vocab.get(token, self.vocab["token"]) for token in text.split()]

    def decode(self, token_ids):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return " ".join(self._inv_vocab.get(t, "token") for t in token_ids)

    def __call__(self, text: str, return_tensors: str = "pt"):
        token_ids = self.encode(text)
        if not token_ids:
            token_ids = [self.pad_token_id]
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return DummyBatch({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        })


class ToyConfig:
    def __init__(self, vocab_size: int = 16, hidden_size: int = 8, num_hidden_layers: int = 2):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = hidden_size
        self.model_type = "toy"


class ToySelfAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = hidden_size
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn_probs = attn_scores.softmax(dim=-1)
        attn_output = torch.matmul(attn_probs, v)
        return self.o_proj(attn_output)


class ToyMLP(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, hidden_size)
        self.up_proj = nn.Linear(hidden_size, hidden_size)
        self.down_proj = nn.Linear(hidden_size, hidden_size)
        self.act = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.act(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return self.down_proj(gate * up)


class ToyBlock(nn.Module):
    def __init__(self, config: ToyConfig):
        super().__init__()
        self.self_attn = ToySelfAttention(config.hidden_size)
        self.mlp = ToyMLP(config.hidden_size)
        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return residual + self.mlp(hidden_states)


class ToyBackbone(nn.Module):
    def __init__(self, config: ToyConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([ToyBlock(config) for _ in range(config.num_hidden_layers)])
        self.final_layernorm = nn.LayerNorm(config.hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.final_layernorm(hidden_states)


class ToyAutoregressiveModel(nn.Module):
    def __init__(self, config: ToyConfig):
        super().__init__()
        self.config = config
        self.model = ToyBackbone(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)
        return SimpleNamespace(logits=logits)


def build_toy_model():
    config = ToyConfig()
    model = ToyAutoregressiveModel(config)
    tokenizer = DummyTokenizer()
    return model, tokenizer


def snapshot_module_weights(kn: KnowledgeNeurons):
    snapshots = {}
    for layer_idx in range(kn.n_layers()):
        layer_snapshot = {}
        for module_key in kn.module_param_attrs:
            module, parameter = kn._get_module_parameter(layer_idx, module_key)
            layer_snapshot[module_key] = parameter.detach().clone()
            bias = getattr(module, "bias", None)
            if bias is not None:
                layer_snapshot[f"{module_key}__bias"] = bias.detach().clone()
        snapshots[layer_idx] = layer_snapshot
    return snapshots


def test_module_param_attrs_include_gate():
    model, tokenizer = build_toy_model()
    kn = KnowledgeNeurons(model, tokenizer)
    assert "mlp_gate" in kn.module_param_attrs


def test_prepare_inputs_truncates_tokens():
    model, tokenizer = build_toy_model()
    kn = KnowledgeNeurons(model, tokenizer, max_target_tokens=3)
    encoded_input = tokenizer("question", return_tensors="pt")
    _, _, target_tokens = kn._prepare_inputs("question", "answer token long response", encoded_input)
    assert len(target_tokens) == 3


def test_module_scores_restore_weights_and_include_gate():
    model, tokenizer = build_toy_model()
    kn = KnowledgeNeurons(model, tokenizer)
    snapshots = snapshot_module_weights(kn)

    scores = kn.get_scores(
        "question",
        "answer",
        steps=2,
        batch_size=1,
        attribution_method="integrated_grads",
        pbar=False,
        target_scope="modules",
    )

    assert set(kn.module_param_attrs).issubset(scores.keys())
    for layer_idx in range(kn.n_layers()):
        for module_key in kn.module_param_attrs:
            module, parameter = kn._get_module_parameter(layer_idx, module_key)
            assert torch.equal(parameter, snapshots[layer_idx][module_key])
            bias = getattr(module, "bias", None)
            if bias is not None:
                assert torch.equal(bias, snapshots[layer_idx][f"{module_key}__bias"])


def test_module_risk_scoring_workflow():
    model, tokenizer = build_toy_model()
    kn = KnowledgeNeurons(model, tokenizer, max_target_tokens=4)
    qa_pairs = [
        ("question", "answer token"),
        ("question", "long response reply token"),
    ]

    results = kn.score_module_risks(
        qa_pairs,
        steps=2,
        batch_size=1,
        max_answer_tokens=2,
        pbar=False,
    )

    assert set(results.keys()) == {"per_prompt", "mean", "max"}
    for collection in results.values():
        for key, tensor in collection.items():
            assert tensor.shape[-1] == kn.n_layers()
            if collection is results["per_prompt"]:
                assert tensor.shape[0] == len(qa_pairs)

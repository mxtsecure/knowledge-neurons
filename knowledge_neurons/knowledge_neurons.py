# main knowledge neurons class
import torch
import torch.nn.functional as F
import torch.nn as nn
import einops
from tqdm import tqdm
import numpy as np
import collections
from typing import Dict, List, Optional, Tuple, Callable, Union
import torch
import torch.nn.functional as F
import einops
import collections
import math
from functools import partial
from transformers import PreTrainedTokenizerBase
from .patch import *


class KnowledgeNeurons:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        device: str = None,
        max_target_tokens: int = 20,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.max_target_tokens = max_target_tokens

        self.model_type = getattr(getattr(self.model, "config", object()), "model_type", "autoregressive")
        if isinstance(self.model_type, str):
            self.model_type = self.model_type.lower()

        self.baseline_activations = None

        (
            self.transformer_layers_attr,
            self.input_ff_attr,
            self.output_ff_attr,
            self.module_param_attrs,
        ) = self._configure_model_specifics()

    def _configure_model_specifics(self):
        """Infer architecture-specific attribute paths for autoregressive transformers."""

        layer_attr_candidates = [
            "model.layers",
            "model.decoder.layers",
            "transformer.h",
            "gpt_neox.layers",
        ]

        transformer_layers = None
        transformer_layers_attr = None
        for attr in layer_attr_candidates:
            try:
                layers = get_attributes(self.model, attr)
            except AttributeError:
                continue
            if hasattr(layers, "__len__") and len(layers) > 0:
                transformer_layers = layers
                transformer_layers_attr = attr
                break
        if transformer_layers is None:
            raise ValueError("Unable to locate transformer layers for the provided model")

        sample_layer = transformer_layers[0]
        named_modules = dict(sample_layer.named_modules())

        module_param_attrs = {}

        module_names = list(named_modules.keys())

        def find_by_suffix(*suffixes):
            for suffix in suffixes:
                for name in module_names:
                    if not name:
                        continue
                    if name.endswith(suffix):
                        return name
            return None

        def find_exact_module(name):
            if name in named_modules:
                return name
            for candidate in module_names:
                if candidate.endswith(name):
                    return candidate
            return None

        input_ff_attr = find_exact_module("mlp") or find_exact_module("feed_forward")
        if input_ff_attr is None:
            raise ValueError("Unable to determine feed-forward module for the provided model")

        mlp_up_attr = find_by_suffix("up_proj", "c_fc", "fc1", "wi_1")
        if mlp_up_attr is not None:
            module_param_attrs["mlp_up"] = mlp_up_attr

        mlp_down_attr = find_by_suffix("down_proj", "c_proj", "fc2", "wo")
        if mlp_down_attr is not None:
            module_param_attrs["mlp_down"] = mlp_down_attr
        else:
            raise ValueError("Unable to locate the feed-forward down projection for the provided model")

        mlp_gate_attr = find_by_suffix("gate_proj", "gating", "wi_0")
        if mlp_gate_attr is not None:
            module_param_attrs["mlp_gate"] = mlp_gate_attr

        qkv_attr = find_by_suffix("c_attn", "attn_fused_qkv")
        if qkv_attr is not None:
            module_param_attrs["qkv_proj"] = qkv_attr
        else:
            q_attr = find_by_suffix("q_proj", "query")
            k_attr = find_by_suffix("k_proj", "key")
            v_attr = find_by_suffix("v_proj", "value")
            if q_attr is not None:
                module_param_attrs["q_proj"] = q_attr
            if k_attr is not None:
                module_param_attrs["k_proj"] = k_attr
            if v_attr is not None:
                module_param_attrs["v_proj"] = v_attr

        o_proj_attr = find_by_suffix("o_proj", "out_proj", "c_proj")
        if o_proj_attr is not None:
            module_param_attrs["o_proj"] = o_proj_attr

        output_ff_attr = f"{mlp_down_attr}.weight"

        if not module_param_attrs:
            raise ValueError("Unable to infer module parameter mappings for the provided model")

        return transformer_layers_attr, input_ff_attr, output_ff_attr, module_param_attrs

    def _get_output_ff_layer(self, layer_idx):
        return get_ff_layer(
            self.model,
            layer_idx,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.output_ff_attr,
        )

    def _get_input_ff_layer(self, layer_idx):
        return get_ff_layer(
            self.model,
            layer_idx,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr,
        )

    def _get_word_embeddings(self):
        if hasattr(self.model, "get_input_embeddings"):
            embeddings = self.model.get_input_embeddings()
            if hasattr(embeddings, "weight"):
                return embeddings.weight
            return embeddings
        if hasattr(self, "word_embeddings_attr") and self.word_embeddings_attr:
            return get_attributes(self.model, self.word_embeddings_attr)
        raise AttributeError("Unable to locate input embedding weights for the provided model")

    def _get_transformer_layers(self):
        return get_attributes(self.model, self.transformer_layers_attr)

    def _get_module_parameter(
        self, layer_idx: int, module_key: str
    ) -> Tuple[nn.Module, nn.Parameter]:
        """Return the module and weight parameter identified by ``module_key`` in ``layer_idx``."""

        if module_key not in self.module_param_attrs:
            raise KeyError(f"Unknown module key: {module_key}")
        transformer_layers = self._get_transformer_layers()
        if layer_idx >= len(transformer_layers):
            raise IndexError(
                f"Layer index {layer_idx} out of range for {len(transformer_layers)} layers"
            )
        module_attr = self.module_param_attrs[module_key]
        module = get_attributes(transformer_layers[layer_idx], module_attr)
        if not isinstance(module, nn.Module):
            raise TypeError(
                f"Attribute '{module_attr}' resolved to a non-module object: {type(module)}"
            )
        if not hasattr(module, "weight"):
            raise AttributeError(
                f"Module '{module_attr}' does not expose a 'weight' parameter"
            )
        parameter = module.weight
        if not isinstance(parameter, nn.Parameter):
            raise TypeError(
                f"Attribute 'weight' of module '{module_attr}' is not an nn.Parameter"
            )
        return module, parameter

    def _prepare_inputs(self, prompt, target=None, encoded_input=None):
        if encoded_input is None:
            encoded_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # with autoregressive models we always want to target the last token
        mask_idx = -1
        if target is not None:
            token_ids = self.tokenizer.encode(target, add_special_tokens=False)
            if self.max_target_tokens is not None:
                token_ids = token_ids[: self.max_target_tokens]
            target = token_ids
        return encoded_input, mask_idx, target

    def _generate(self, prompt, ground_truth):
        encoded_input, mask_idx, target_label = self._prepare_inputs(
            prompt, ground_truth
        )
        # for autoregressive models, we might want to generate > 1 token
        n_sampling_steps = (
            len(target_label) if isinstance(target_label, (list, tuple)) else 1
        )

        all_gt_probs = []
        all_argmax_probs = []
        argmax_tokens = []
        argmax_completion_str = ""

        for i in range(n_sampling_steps):
            if i > 0:
                # retokenize new inputs
                encoded_input, mask_idx, target_label = self._prepare_inputs(
                    prompt, ground_truth
                )
            outputs = self.model(**encoded_input)
            probs = F.softmax(outputs.logits[:, mask_idx, :], dim=-1)
            if n_sampling_steps > 1:
                target_idx = target_label[i]
            else:
                target_idx = target_label
            gt_prob = probs[:, target_idx].item()
            all_gt_probs.append(gt_prob)

            # get info about argmax completion
            argmax_prob, argmax_id = [i.item() for i in probs.max(dim=-1)]
            argmax_tokens.append(argmax_id)
            argmax_str = self.tokenizer.decode([argmax_id])
            all_argmax_probs.append(argmax_prob)

            prompt += argmax_str
            argmax_completion_str += argmax_str

        gt_prob = math.prod(all_gt_probs) if len(all_gt_probs) > 1 else all_gt_probs[0]
        argmax_prob = (
            math.prod(all_argmax_probs)
            if len(all_argmax_probs) > 1
            else all_argmax_probs[0]
        )
        return gt_prob, argmax_prob, argmax_completion_str, argmax_tokens

    def n_layers(self):
        return len(self._get_transformer_layers())

    def intermediate_size(self):
        if hasattr(self.model.config, "intermediate_size"):
            return self.model.config.intermediate_size
        if hasattr(self.model.config, "hidden_size"):
            return self.model.config.hidden_size * 4
        raise AttributeError("Model configuration does not expose an intermediate size")

    @staticmethod
    def scaled_input(activations: torch.Tensor, steps: int = 20, device: str = "cpu"):
        """
        Tiles activations along the batch dimension - gradually scaling them over
        `steps` steps from 0 to their original value over the batch dimensions.

        `activations`: torch.Tensor
        original activations
        `steps`: int
        number of steps to take
        """
        tiled_activations = einops.repeat(activations, "b d -> (r b) d", r=steps)
        out = (
            tiled_activations
            * torch.linspace(start=0, end=1, steps=steps).to(device)[:, None]
        )
        return out

    def get_baseline_with_activations(
        self, encoded_input: dict, layer_idx: int, mask_idx: int
    ):
        """
        Gets the baseline outputs and activations for the unmodified model at a given index.

        `encoded_input`: torch.Tensor
            the inputs to the model from self.tokenizer.encode_plus()
        `layer_idx`: int
            which transformer layer to access
        `mask_idx`: int
            the position at which to get the activations (TODO: rename? with autoregressive models there's no mask, so)
        """

        def get_activations(model, layer_idx, mask_idx):
            """
            This hook function should assign the intermediate activations at a given layer / mask idx
            to the 'self.baseline_activations' variable
            """

            def hook_fn(acts):
                self.baseline_activations = acts[:, mask_idx, :]

            return register_hook(
                model,
                layer_idx=layer_idx,
                f=hook_fn,
                transformer_layers_attr=self.transformer_layers_attr,
                ff_attrs=self.input_ff_attr,
            )

        handle = get_activations(self.model, layer_idx=layer_idx, mask_idx=mask_idx)
        baseline_outputs = self.model(**encoded_input)
        handle.remove()
        baseline_activations = self.baseline_activations
        self.baseline_activations = None
        return baseline_outputs, baseline_activations

    def _integrated_grads_for_module(
        self,
        encoded_input: Dict[str, torch.Tensor],
        layer_idx: int,
        module_key: str,
        mask_idx: int,
        target_idx: int,
        steps: int,
    ) -> torch.Tensor:
        """Estimate the integrated gradient for ``module_key`` in ``layer_idx``."""

        module, parameter = self._get_module_parameter(layer_idx, module_key)
        original_weight = parameter.detach().clone()
        baseline_weight = torch.zeros_like(original_weight)
        delta_weight = original_weight - baseline_weight

        bias = getattr(module, "bias", None)
        original_bias = bias.detach().clone() if bias is not None else None

        ig_value = torch.zeros((), device=parameter.device, dtype=parameter.dtype)
        alphas = torch.linspace(0.0, 1.0, steps, device=parameter.device)

        for alpha in alphas:
            scaled_weight = baseline_weight + alpha * delta_weight
            parameter.data.copy_(scaled_weight)
            parameter.requires_grad_(True)
            if bias is not None and original_bias is not None:
                bias.data.copy_(original_bias)
                bias.requires_grad_(True)

            self.model.zero_grad(set_to_none=True)
            outputs = self.model(**encoded_input)
            probs = F.softmax(outputs.logits[:, mask_idx, :], dim=-1)
            target_prob = probs[:, target_idx].sum()
            grad = torch.autograd.grad(target_prob, parameter, retain_graph=False)[0]
            ig_value += (grad * delta_weight).sum() / steps

            parameter.data.copy_(original_weight)
            if bias is not None and original_bias is not None:
                bias.data.copy_(original_bias)

        parameter.data.copy_(original_weight)
        if bias is not None and original_bias is not None:
            bias.data.copy_(original_bias)
        return ig_value.detach()

    def get_scores(
        self,
        prompt: str,
        ground_truth: str,
        batch_size: int = 10,
        steps: int = 20,
        attribution_method: str = "integrated_grads",
        pbar: bool = True,
        target_scope: str = "neurons",
    ):
        """
        Gets the attribution scores for a given prompt and ground truth.
        `prompt`: str
            the prompt to get the attribution scores for
        `ground_truth`: str
            the ground truth / expected output
        `batch_size`: int
            batch size
        `steps`: int
            total number of steps (per token) for the integrated gradient calculations
        `attribution_method`: str
            the method to use for getting the scores. Choose from 'integrated_grads' or 'max_activations'.
        `target_scope`: str
            whether to attribute activations ("neurons") or full module parameters ("modules").
        """

        if target_scope not in {"neurons", "modules"}:
            raise ValueError(f"Unknown target scope: {target_scope}")
        if target_scope == "modules" and not self.module_param_attrs:
            raise ValueError("Module parameter mappings are not defined for this model")
        scores = []
        module_scores: Dict[str, List[torch.Tensor]] = (
            {k: [] for k in self.module_param_attrs}
            if target_scope == "modules"
            else {}
        )
        encoded_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        for layer_idx in tqdm(
            range(self.n_layers()),
            desc="Getting attribution scores for each layer...",
            disable=not pbar,
        ):
            layer_scores = self.get_scores_for_layer(
                prompt,
                ground_truth,
                encoded_input=encoded_input,
                layer_idx=layer_idx,
                batch_size=batch_size,
                steps=steps,
                attribution_method=attribution_method,
                target_scope=target_scope,
            )
            if target_scope == "modules":
                for module_key in self.module_param_attrs:
                    module_scores[module_key].append(layer_scores[module_key])
            else:
                scores.append(layer_scores)
        if target_scope == "modules":
            return {
                module_key: torch.stack(module_scores[module_key])
                for module_key in self.module_param_attrs
            }
        return torch.stack(scores)

    def score_module_risks(
        self,
        qa_pairs: List[Tuple[str, str]],
        *,
        steps: int = 20,
        batch_size: int = 10,
        max_answer_tokens: Optional[int] = None,
        pbar: bool = False,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Compute aggregated module-level attribution scores for QA datasets."""

        if not qa_pairs:
            raise ValueError("Expected at least one question/answer pair")

        original_max_tokens = self.max_target_tokens
        if max_answer_tokens is not None:
            self.max_target_tokens = max_answer_tokens

        per_prompt_scores: Dict[str, List[torch.Tensor]] = {
            key: [] for key in self.module_param_attrs
        }

        try:
            for prompt, answer in tqdm(
                qa_pairs,
                desc="Scoring module risks...",
                disable=not pbar,
            ):
                module_scores = self.get_scores(
                    prompt,
                    answer,
                    steps=steps,
                    batch_size=batch_size,
                    attribution_method="integrated_grads",
                    pbar=False,
                    target_scope="modules",
                )
                for key, value in module_scores.items():
                    per_prompt_scores[key].append(value.detach().cpu())
        finally:
            self.max_target_tokens = original_max_tokens

        stacked_scores: Dict[str, torch.Tensor] = {}
        for key, values in per_prompt_scores.items():
            if values:
                stacked_scores[key] = torch.stack(values)
            else:
                stacked_scores[key] = torch.empty(
                    (0, self.n_layers()), dtype=torch.float32
                )

        mean_scores = {
            key: values.mean(dim=0) if values.numel() > 0 else values
            for key, values in stacked_scores.items()
        }
        max_scores = {
            key: values.max(dim=0).values if values.numel() > 0 else values
            for key, values in stacked_scores.items()
        }

        return {
            "per_prompt": stacked_scores,
            "mean": mean_scores,
            "max": max_scores,
        }

    def get_coarse_neurons(
        self,
        prompt: str,
        ground_truth: str,
        batch_size: int = 10,
        steps: int = 20,
        threshold: float = None,
        adaptive_threshold: float = None,
        percentile: float = None,
        attribution_method: str = "integrated_grads",
        pbar: bool = True,
        target_scope: str = "neurons",
    ) -> List[List[Union[int, str]]]:
        """
        Finds the 'coarse' neurons for a given prompt and ground truth.
        The coarse neurons are the neurons that are most activated by a single prompt.
        We refine these by using multiple prompts that express the same 'fact'/relation in different ways.

        `prompt`: str
            the prompt to get the coarse neurons for
        `ground_truth`: str
            the ground truth / expected output
        `batch_size`: int
            batch size
        `steps`: int
            total number of steps (per token) for the integrated gradient calculations
        `threshold`: float
            `t` from the paper. If not None, then we only keep neurons with integrated grads above this threshold.
        `adaptive_threshold`: float
            Adaptively set `threshold` based on `maximum attribution score * adaptive_threshold` (in the paper, they set adaptive_threshold=0.3)
        `percentile`: float
            If not None, then we only keep neurons with integrated grads in this percentile of all integrated grads.
        `attribution_method`: str
            the method to use for getting the scores. Choose from 'integrated_grads' or 'max_activations'.
        `target_scope`: str
            whether to return individual neuron indices ("neurons") or module identifiers ("modules").
        """
        attribution_scores = self.get_scores(
            prompt,
            ground_truth,
            batch_size=batch_size,
            steps=steps,
            pbar=pbar,
            attribution_method=attribution_method,
            target_scope=target_scope,
        )
        assert (
            sum(e is not None for e in [threshold, adaptive_threshold, percentile]) == 1
        ), f"Provide one and only one of threshold / adaptive_threshold / percentile"
        if adaptive_threshold is not None:
            if target_scope == "modules":
                stacked_scores = torch.stack(
                    [attribution_scores[k] for k in self.module_param_attrs], dim=-1
                )
                threshold = (
                    stacked_scores.max().item() * adaptive_threshold
                    if stacked_scores.numel() > 0
                    else 0.0
                )
            else:
                threshold = attribution_scores.max().item() * adaptive_threshold
        if target_scope == "modules":
            # Interpret the attribution tensor as layer-wise module scores rather than neuron activations.
            stacked_scores = torch.stack(
                [attribution_scores[k] for k in self.module_param_attrs], dim=-1
            )
            if threshold is not None:
                mask = stacked_scores > threshold
            else:
                s = stacked_scores.flatten().detach().cpu().numpy()
                mask = stacked_scores > np.percentile(s, percentile)
            indices = torch.nonzero(mask).cpu().tolist()
            module_names = list(self.module_param_attrs.keys())
            return [[layer_idx, module_names[module_idx]] for layer_idx, module_idx in indices]
        if threshold is not None:
            return torch.nonzero(attribution_scores > threshold).cpu().tolist()
        s = attribution_scores.flatten().detach().cpu().numpy()
        return (
            torch.nonzero(attribution_scores > np.percentile(s, percentile))
            .cpu()
            .tolist()
        )

    def get_refined_neurons(
        self,
        prompts: List[str],
        ground_truth: str,
        negative_examples: Optional[List[str]] = None,
        p: float = 0.5,
        batch_size: int = 10,
        steps: int = 20,
        coarse_adaptive_threshold: Optional[float] = 0.3,
        coarse_threshold: Optional[float] = None,
        coarse_percentile: Optional[float] = None,
        quiet=False,
        target_scope: str = "neurons",
    ) -> List[List[Union[int, str]]]:
        """
        Finds the 'refined' neurons for a given set of prompts and a ground truth / expected output.

        The input should be n different prompts, each expressing the same fact in different ways.
        For each prompt, we calculate the attribution scores of each intermediate neuron.
        We then set an attribution score threshold, and we keep the neurons that are above this threshold.
        Finally, considering the coarse neurons from all prompts, we set a sharing percentage threshold, p,
        and retain only neurons shared by more than p% of prompts.

        `prompts`: list of str
            the prompts to get the refined neurons for
        `ground_truth`: str
            the ground truth / expected output
        `negative_examples`: list of str
            Optionally provide a list of negative examples. Any neuron that appears in these examples will be excluded from the final results.
        `p`: float
            the threshold for the sharing percentage
        `batch_size`: int
            batch size
        `steps`: int
            total number of steps (per token) for the integrated gradient calculations
        `coarse_threshold`: float
            threshold for the coarse neurons
        `coarse_percentile`: float
            percentile for the coarse neurons
        `target_scope`: str
            whether to combine neuron-level or module-level identifiers.
        """
        assert isinstance(
            prompts, list
        ), "Must provide a list of different prompts to get refined neurons"
        assert 0.0 <= p < 1.0, "p should be a float between 0 and 1"

        n_prompts = len(prompts)
        coarse_neurons = []
        for prompt in tqdm(
            prompts, desc="Getting coarse neurons for each prompt...", disable=quiet
        ):
            coarse_neurons.append(
                self.get_coarse_neurons(
                    prompt,
                    ground_truth,
                    batch_size=batch_size,
                    steps=steps,
                    adaptive_threshold=coarse_adaptive_threshold,
                    threshold=coarse_threshold,
                    percentile=coarse_percentile,
                    pbar=False,
                    target_scope=target_scope,
                )
            )
        if negative_examples is not None:
            negative_neurons = []
            for negative_example in tqdm(
                negative_examples,
                desc="Getting coarse neurons for negative examples",
                disable=quiet,
            ):
                negative_neurons.append(
                    self.get_coarse_neurons(
                        negative_example,
                        ground_truth,
                        batch_size=batch_size,
                        steps=steps,
                        adaptive_threshold=coarse_adaptive_threshold,
                        threshold=coarse_threshold,
                        percentile=coarse_percentile,
                        pbar=False,
                        target_scope=target_scope,
                    )
                )
        if not quiet:
            total_coarse_neurons = sum([len(i) for i in coarse_neurons])
            print(f"\n{total_coarse_neurons} coarse neurons found - refining")
        t = n_prompts * p
        refined_neurons = []
        c = collections.Counter()
        for neurons in coarse_neurons:
            for n in neurons:
                c[tuple(n)] += 1

        for neuron, count in c.items():
            if count > t:
                refined_neurons.append(list(neuron))

        # filter out neurons that are in the negative examples
        if negative_examples is not None:
            for neuron in negative_neurons:
                if neuron in refined_neurons:
                    refined_neurons.remove(neuron)

        total_refined_neurons = len(refined_neurons)
        if not quiet:
            print(f"{total_refined_neurons} attribution entries remaining after refining")
        return refined_neurons

    def get_scores_for_layer(
        self,
        prompt: str,
        ground_truth: str,
        layer_idx: int,
        batch_size: int = 10,
        steps: int = 20,
        encoded_input: Optional[int] = None,
        attribution_method: str = "integrated_grads",
        target_scope: str = "neurons",
    ):
        """
        get the attribution scores for a given layer
        `prompt`: str
            the prompt to get the attribution scores for
        `ground_truth`: str
            the ground truth / expected output
        `layer_idx`: int
            the layer to get the scores for
        `batch_size`: int
            batch size
        `steps`: int
            total number of steps (per token) for the integrated gradient calculations
        `encoded_input`: int
            if not None, then use this encoded input instead of getting a new one
        `attribution_method`: str
            the method to use for getting the scores. Choose from 'integrated_grads' or 'max_activations'.
        `target_scope`: str
            whether to attribute activations ("neurons") or module parameters ("modules").
        """
        if target_scope not in {"neurons", "modules"}:
            raise ValueError(f"Unknown target scope: {target_scope}")
        if target_scope == "neurons":
            assert steps % batch_size == 0
            n_batches = steps // batch_size
        else:
            n_batches = None

        # First we take the unmodified model and use a hook to return the baseline intermediate activations at our chosen target layer
        encoded_input, mask_idx, target_label = self._prepare_inputs(
            prompt, ground_truth, encoded_input
        )

        # for autoregressive models, we might want to generate > 1 token
        n_sampling_steps = (
            len(target_label) if isinstance(target_label, (list, tuple)) else 1
        )

        if attribution_method == "integrated_grads":
            if target_scope == "modules":
                module_scores = {
                    module_key: torch.zeros((), device=self.device)
                    for module_key in self.module_param_attrs
                }

                for i in range(n_sampling_steps):
                    if i > 0 and n_sampling_steps > 1:
                        encoded_input, mask_idx, target_label = self._prepare_inputs(
                            prompt, ground_truth
                        )
                    baseline_outputs, _ = self.get_baseline_with_activations(
                        encoded_input, layer_idx, mask_idx
                    )
                    if n_sampling_steps > 1:
                        argmax_next_token = (
                            baseline_outputs.logits[:, mask_idx, :]
                            .argmax(dim=-1)
                            .item()
                        )
                        next_token_str = self.tokenizer.decode(argmax_next_token)

                    current_target = target_label[i] if n_sampling_steps > 1 else target_label
                    for module_key in self.module_param_attrs:
                        module_scores[module_key] += self._integrated_grads_for_module(
                            encoded_input,
                            layer_idx,
                            module_key,
                            mask_idx,
                            current_target,
                            steps,
                        )
                    if n_sampling_steps > 1:
                        prompt += next_token_str

                return {
                    module_key: score / n_sampling_steps
                    for module_key, score in module_scores.items()
                }

            integrated_grads = []

            for i in range(n_sampling_steps):
                if i > 0 and n_sampling_steps > 1:
                    encoded_input, mask_idx, target_label = self._prepare_inputs(
                        prompt, ground_truth
                    )
                (
                    baseline_outputs,
                    baseline_activations,
                ) = self.get_baseline_with_activations(
                    encoded_input, layer_idx, mask_idx
                )
                if n_sampling_steps > 1:
                    argmax_next_token = (
                        baseline_outputs.logits[:, mask_idx, :].argmax(dim=-1).item()
                    )
                    next_token_str = self.tokenizer.decode(argmax_next_token)

                # Now we want to gradually change the intermediate activations of our layer from 0 -> their original value
                # and calculate the integrated gradient of the masked position at each step
                # we do this by repeating the input across the batch dimension, multiplying the first batch by 0, the second by 0.1, etc., until we reach 1
                scaled_weights = self.scaled_input(
                    baseline_activations, steps=steps, device=self.device
                )
                scaled_weights.requires_grad_(True)

                integrated_grads_this_step = []  # to store the integrated gradients

                for batch_weights in scaled_weights.chunk(n_batches):
                    # we want to replace the intermediate activations at some layer, at the mask position, with `batch_weights`
                    # first tile the inputs to the correct batch size
                    inputs = {
                        "input_ids": einops.repeat(
                            encoded_input["input_ids"], "b d -> (r b) d", r=batch_size
                        ),
                        "attention_mask": einops.repeat(
                            encoded_input.get("attention_mask", torch.ones_like(encoded_input["input_ids"])),
                            "b d -> (r b) d",
                            r=batch_size,
                        ),
                    }

                    # then patch the model to replace the activations with the scaled activations
                    patch_ff_layer(
                        self.model,
                        layer_idx=layer_idx,
                        mask_idx=mask_idx,
                        replacement_activations=batch_weights,
                        transformer_layers_attr=self.transformer_layers_attr,
                        ff_attrs=self.input_ff_attr,
                    )

                    # then forward through the model to get the logits
                    outputs = self.model(**inputs)

                    # then calculate the gradients for each step w/r/t the inputs
                    probs = F.softmax(outputs.logits[:, mask_idx, :], dim=-1)
                    if n_sampling_steps > 1:
                        target_idx = target_label[i]
                    else:
                        target_idx = target_label
                    grad = torch.autograd.grad(
                        torch.unbind(probs[:, target_idx]), batch_weights
                    )[0]
                    grad = grad.sum(dim=0)
                    integrated_grads_this_step.append(grad)

                    unpatch_ff_layer(
                        self.model,
                        layer_idx=layer_idx,
                        transformer_layers_attr=self.transformer_layers_attr,
                        ff_attrs=self.input_ff_attr,
                    )

                # then sum, and multiply by W-hat / m
                integrated_grads_this_step = torch.stack(
                    integrated_grads_this_step, dim=0
                ).sum(dim=0)
                integrated_grads_this_step *= baseline_activations.squeeze(0) / steps
                integrated_grads.append(integrated_grads_this_step)

                if n_sampling_steps > 1:
                    prompt += next_token_str
            integrated_grads = torch.stack(integrated_grads, dim=0).sum(dim=0) / len(
                integrated_grads
            )
            return integrated_grads
        elif attribution_method == "max_activations":
            if target_scope != "neurons":
                raise NotImplementedError(
                    "Module scope is only supported for integrated gradients"
                )
            activations = []
            for i in range(n_sampling_steps):
                if i > 0 and n_sampling_steps > 1:
                    encoded_input, mask_idx, target_label = self._prepare_inputs(
                        prompt, ground_truth
                    )
                (
                    baseline_outputs,
                    baseline_activations,
                ) = self.get_baseline_with_activations(
                    encoded_input, layer_idx, mask_idx
                )
                activations.append(baseline_activations)
                if n_sampling_steps > 1:
                    argmax_next_token = (
                        baseline_outputs.logits[:, mask_idx, :].argmax(dim=-1).item()
                    )
                    next_token_str = self.tokenizer.decode(argmax_next_token)
                    prompt += next_token_str
            activations = torch.stack(activations, dim=0).sum(dim=0) / len(activations)
            return activations.squeeze(0)
        else:
            raise NotImplementedError

    def modify_activations(
        self,
        prompt: str,
        ground_truth: str,
        neurons: List[List[int]],
        mode: str = "suppress",
        undo_modification: bool = True,
        quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        results_dict = {}
        _, mask_idx, _ = self._prepare_inputs(
            prompt, ground_truth
        )  # just need to get the mask index for later - probably a better way to do this
        # get the baseline probabilities of the groundtruth being generated + the argmax / greedy completion before modifying the activations
        (
            gt_baseline_prob,
            argmax_baseline_prob,
            argmax_completion_str,
            _,
        ) = self._generate(prompt, ground_truth)
        if not quiet:
            print(
                f"\nBefore modification - groundtruth probability: {gt_baseline_prob}\nArgmax completion: `{argmax_completion_str}`\nArgmax prob: {argmax_baseline_prob}\n"
            )
        results_dict["before"] = {
            "gt_prob": gt_baseline_prob,
            "argmax_completion": argmax_completion_str,
            "argmax_prob": argmax_baseline_prob,
        }

        # patch model to suppress neurons
        # store all the layers we patch so we can unpatch them later
        all_layers = set([n[0] for n in neurons])

        patch_ff_layer(
            self.model,
            mask_idx,
            mode=mode,
            neurons=neurons,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr,
        )

        # get the probabilities of the groundtruth being generated + the argmax / greedy completion after modifying the activations
        new_gt_prob, new_argmax_prob, new_argmax_completion_str, _ = self._generate(
            prompt, ground_truth
        )
        if not quiet:
            print(
                f"\nAfter modification - groundtruth probability: {new_gt_prob}\nArgmax completion: `{new_argmax_completion_str}`\nArgmax prob: {new_argmax_prob}\n"
            )
        results_dict["after"] = {
            "gt_prob": new_gt_prob,
            "argmax_completion": new_argmax_completion_str,
            "argmax_prob": new_argmax_prob,
        }

        unpatch_fn = partial(
            unpatch_ff_layers,
            model=self.model,
            layer_indices=all_layers,
            transformer_layers_attr=self.transformer_layers_attr,
            ff_attrs=self.input_ff_attr,
        )

        if undo_modification:
            unpatch_fn()
            unpatch_fn = lambda *args: args

        return results_dict, unpatch_fn

    def suppress_knowledge(
        self,
        prompt: str,
        ground_truth: str,
        neurons: List[List[int]],
        undo_modification: bool = True,
        quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        """
        prompt the model with `prompt`, zeroing the activations at the positions specified by `neurons`,
        and measure the resulting affect on the ground truth probability.
        """
        return self.modify_activations(
            prompt=prompt,
            ground_truth=ground_truth,
            neurons=neurons,
            mode="suppress",
            undo_modification=undo_modification,
            quiet=quiet,
        )

    def enhance_knowledge(
        self,
        prompt: str,
        ground_truth: str,
        neurons: List[List[int]],
        undo_modification: bool = True,
        quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        """
        prompt the model with `prompt`, multiplying the activations at the positions
        specified by `neurons` by 2, and measure the resulting affect on the ground truth probability.
        """
        return self.modify_activations(
            prompt=prompt,
            ground_truth=ground_truth,
            neurons=neurons,
            mode="enhance",
            undo_modification=undo_modification,
            quiet=quiet,
        )

    @torch.no_grad()
    def modify_weights(
        self,
        prompt: str,
        neurons: List[List[int]],
        target: str,
        mode: str = "edit",
        erase_value: str = "zero",
        undo_modification: bool = True,
        quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        raise NotImplementedError(
            "Weight editing workflows are not supported for autoregressive models"
        )

    def edit_knowledge(
        self,
        prompt: str,
        target: str,
        neurons: List[List[int]],
        undo_modification: bool = True,
        quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        return self.modify_weights(
            prompt=prompt,
            neurons=neurons,
            target=target,
            mode="edit",
            undo_modification=undo_modification,
            quiet=quiet,
        )

    def erase_knowledge(
        self,
        prompt: str,
        neurons: List[List[int]],
        erase_value: str = "zero",
        target: Optional[str] = None,
        undo_modification: bool = True,
        quiet: bool = False,
    ) -> Tuple[dict, Callable]:
        return self.modify_weights(
            prompt=prompt,
            neurons=neurons,
            target=target,
            mode="erase",
            erase_value=erase_value,
            undo_modification=undo_modification,
            quiet=quiet,
        )

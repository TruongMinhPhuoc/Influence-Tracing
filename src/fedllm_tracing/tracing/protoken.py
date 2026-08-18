"""PEFT-compatible ProToken tracing with exact global layer inputs."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from fedllm_tracing.adapters.lora_utils import use_lora_update
from fedllm_tracing.federated.types import ClientUpdate
from fedllm_tracing.tracing.hooks import GlobalForwardTrace, LayerCallContext
from fedllm_tracing.tracing.interfaces import TokenAttribution, TokenTracer


def _tensor_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError("Selected layer output is not a tensor or tensor-first tuple")


class PeftTokenTracer(TokenTracer):
    """Trace PEFT LoRA client adapters without independent client full forwards."""

    def __init__(self, tokenizer: Any | None = None, *, adapter_name: str = "default"):
        self.tokenizer = tokenizer
        self.adapter_name = adapter_name

    def capture_global_states(
        self,
        global_model: torch.nn.Module,
        model_inputs: Mapping[str, torch.Tensor],
        layers: Sequence[str],
    ) -> GlobalForwardTrace:
        if not layers or len(set(layers)) != len(layers):
            raise ValueError("layers must be a non-empty sequence of unique module names")
        pending: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}
        captured: dict[str, LayerCallContext] = {}
        handles: list[Any] = []

        def make_pre_hook(layer_name: str):
            def hook(
                _module: torch.nn.Module,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
            ) -> None:
                pending[layer_name] = (args, dict(kwargs))

            return hook

        def make_post_hook(layer_name: str):
            def hook(
                _module: torch.nn.Module,
                _args: tuple[Any, ...],
                _kwargs: dict[str, Any],
                output: Any,
            ) -> None:
                if layer_name not in pending:
                    raise RuntimeError(f"No captured input for layer {layer_name!r}")
                args, kwargs = pending[layer_name]
                captured[layer_name] = LayerCallContext(
                    args=args,
                    kwargs=kwargs,
                    output=_tensor_output(output),
                )

            return hook

        for layer_name in layers:
            module = global_model.get_submodule(layer_name)
            handles.append(
                module.register_forward_pre_hook(
                    make_pre_hook(layer_name), with_kwargs=True
                )
            )
            handles.append(
                module.register_forward_hook(
                    make_post_hook(layer_name), with_kwargs=True
                )
            )
        try:
            output = global_model(**model_inputs, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        missing = set(layers) - set(captured)
        if missing:
            raise RuntimeError(f"Selected layers were not executed: {sorted(missing)}")
        return GlobalForwardTrace(output, captured)

    def compute_client_layer_activation(
        self,
        global_model: torch.nn.Module,
        client_update: ClientUpdate,
        layer: str,
        global_layer_call: LayerCallContext,
    ) -> torch.Tensor:
        module = global_model.get_submodule(layer)
        with use_lora_update(
            global_model, client_update, adapter_name=self.adapter_name
        ), torch.no_grad():
            output = module(*global_layer_call.args, **global_layer_call.kwargs)
        return _tensor_output(output)

    def compute_token_gradient(
        self,
        target_logit: torch.Tensor,
        layer_output: torch.Tensor,
        *,
        retain_graph: bool = True,
    ) -> torch.Tensor:
        gradient = torch.autograd.grad(
            target_logit,
            layer_output,
            retain_graph=retain_graph,
            create_graph=False,
            allow_unused=False,
        )[0]
        return gradient

    def compute_attribution_score(
        self, client_activation: torch.Tensor, token_gradient: torch.Tensor
    ) -> float:
        if client_activation.shape != token_gradient.shape:
            raise ValueError(
                "Client activation and token gradient must have identical shapes"
            )
        return float(
            torch.sum(client_activation.detach().float() * token_gradient.detach().float())
            .cpu()
            .item()
        )

    def trace_token_ids(
        self,
        global_model: torch.nn.Module,
        client_updates: Sequence[ClientUpdate],
        prompt_ids: Sequence[int],
        target_tokens: Sequence[int],
        layers: Sequence[str],
    ) -> list[TokenAttribution]:
        """Teacher-force target IDs and return one score per client and token."""
        if not client_updates:
            raise ValueError("At least one client update is required")
        if not prompt_ids or not target_tokens:
            raise ValueError("prompt_ids and target_tokens must both be non-empty")
        if len({update.client_id for update in client_updates}) != len(client_updates):
            raise ValueError("Client IDs must be unique")
        device = next(global_model.parameters()).device
        all_ids = [int(token) for token in (*prompt_ids, *target_tokens)]
        model_inputs = {
            "input_ids": torch.tensor([all_ids], dtype=torch.long, device=device),
            "attention_mask": torch.ones(
                (1, len(all_ids)), dtype=torch.long, device=device
            ),
        }
        was_training = global_model.training
        global_model.eval()
        try:
            global_trace = self.capture_global_states(
                global_model, model_inputs, layers
            )
            logits = global_trace.model_output.logits
            gradients: dict[tuple[int, str], torch.Tensor] = {}
            for token_index, token_id in enumerate(target_tokens):
                prediction_position = len(prompt_ids) - 1 + token_index
                target_logit = logits[0, prediction_position, int(token_id)]
                for layer in layers:
                    full_gradient = self.compute_token_gradient(
                        target_logit,
                        global_trace.layers[layer].output,
                        retain_graph=True,
                    )
                    gradients[(token_index, layer)] = full_gradient[
                        0, prediction_position
                    ].detach()

            attributions: list[TokenAttribution] = []
            for update in client_updates:
                # Swap once per client, then replay only selected layers. There is
                # never a client-specific full-model forward pass.
                with use_lora_update(
                    global_model, update, adapter_name=self.adapter_name
                ), torch.no_grad():
                    client_outputs = {
                        layer: _tensor_output(
                            global_model.get_submodule(layer)(
                                *global_trace.layers[layer].args,
                                **global_trace.layers[layer].kwargs,
                            )
                        )
                        for layer in layers
                    }
                for token_index, token_id in enumerate(target_tokens):
                    prediction_position = len(prompt_ids) - 1 + token_index
                    layer_scores = {
                        layer: self.compute_attribution_score(
                            client_outputs[layer][0, prediction_position],
                            gradients[(token_index, layer)],
                        )
                        for layer in layers
                    }
                    attributions.append(
                        TokenAttribution(
                            client_id=update.client_id,
                            token_id=int(token_id),
                            layer_scores=layer_scores,
                            token_index=token_index,
                            sequence_position=prediction_position,
                        )
                    )
            return attributions
        finally:
            global_model.train(was_training)

    def trace(
        self,
        global_model: torch.nn.Module,
        client_updates: Sequence[ClientUpdate],
        prompt: str,
        target_tokens: Sequence[int],
        layers: Sequence[str],
        *,
        teacher_forcing: bool = True,
    ) -> list[TokenAttribution]:
        if not teacher_forcing:
            raise NotImplementedError("Autoregressive generation tracing is not implemented")
        if self.tokenizer is None:
            raise ValueError("A tokenizer is required when tracing a text prompt")
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        return self.trace_token_ids(
            global_model, client_updates, prompt_ids, target_tokens, layers
        )


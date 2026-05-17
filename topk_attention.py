"""Inference-time top-k masking utilities for Concept Transformer attention.

This module keeps the existing dense model code untouched and provides a
separate sparse attention path that can be attached at evaluation time.
"""

from copy import deepcopy

import torch

from ctc.ctc import CrossAttention


def topk_mask_from_scores(scores: torch.Tensor, top_k: int) -> torch.Tensor:
    """Return a boolean mask that keeps the top-k entries along the last axis."""

    if top_k is None or top_k <= 0 or top_k >= scores.size(-1):
        return torch.ones_like(scores, dtype=torch.bool)

    topk_indices = scores.topk(top_k, dim=-1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(-1, topk_indices, True)
    return mask


class TopKCrossAttention(CrossAttention):
    """Cross-attention with top-k masking applied before the softmax."""

    def __init__(
        self,
        dim,
        n_outputs=None,
        num_heads=8,
        attention_dropout=0.1,
        projection_dropout=0.0,
        top_k=1,
    ):
        super().__init__(
            dim=dim,
            n_outputs=n_outputs,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            projection_dropout=projection_dropout,
        )
        self.top_k = top_k

    @classmethod
    def from_dense(cls, dense_attention: CrossAttention, top_k: int):
        """Create a top-k attention block from an existing dense attention block."""

        topk_attention = cls(
            dim=dense_attention.q.in_features,
            n_outputs=dense_attention.proj.out_features,
            num_heads=dense_attention.num_heads,
            attention_dropout=dense_attention.attn_drop.p,
            projection_dropout=dense_attention.proj_drop.p,
            top_k=top_k,
        )
        topk_attention.load_state_dict(dense_attention.state_dict())
        return topk_attention

    def forward(self, x, y):
        batch_size, n_queries, channels = x.shape
        batch_size_y, n_keys, channels_y = y.shape

        assert channels == channels_y, "Feature size of x and y must be the same"

        q = self.q(x).reshape(batch_size, n_queries, 1, self.num_heads, channels // self.num_heads)
        q = q.permute(2, 0, 3, 1, 4)
        kv = (
            self.kv(y)
            .reshape(batch_size_y, n_keys, 2, self.num_heads, channels // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        q = q[0]
        k, v = kv[0], kv[1]

        scores = (q @ k.transpose(-2, -1)) * self.scale

        if self.top_k is not None and 0 < self.top_k < scores.size(-1):
            shared_mask = topk_mask_from_scores(scores.mean(dim=1), self.top_k)
            scores = scores.masked_fill(~shared_mask.unsqueeze(1), float("-inf"))

        attn = scores.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, n_queries, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


def patch_concept_transformer_for_topk(concept_transformer, top_k: int):
    """Replace every cross-attention block inside a concept transformer."""

    for attr_name in (
        "unsup_concept_tranformer",
        "concept_tranformer",
        "spatial_concept_tranformer",
    ):
        attention_block = getattr(concept_transformer, attr_name, None)
        if attention_block is not None:
            setattr(
                concept_transformer,
                attr_name,
                TopKCrossAttention.from_dense(attention_block, top_k),
            )
    return concept_transformer


def patch_ctc_model_for_topk(lightning_model, top_k: int):
    """Return a deep-copied Lightning model with sparse inference enabled."""

    patched_model = deepcopy(lightning_model)
    if hasattr(patched_model.model, "concept_transformer"):
        patch_concept_transformer_for_topk(patched_model.model.concept_transformer, top_k)
    if hasattr(patched_model.model, "classifier"):
        patch_concept_transformer_for_topk(patched_model.model.classifier, top_k)
    return patched_model

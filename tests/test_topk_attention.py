import torch

from topk_attention import TopKCrossAttention, topk_mask_from_scores


def test_topk_mask_from_scores_selects_exact_entries():
    scores = torch.tensor([[0.1, 0.9, 0.2, 0.8]])
    mask = topk_mask_from_scores(scores, top_k=2)

    assert mask.tolist() == [[False, True, False, True]]


def test_topk_cross_attention_preserves_shapes_and_sparsity():
    attention = TopKCrossAttention(dim=8, n_outputs=4, num_heads=2, top_k=2)
    attention.eval()
    x = torch.randn(3, 1, 8)
    y = torch.randn(3, 5, 8)

    output, attn = attention(x, y)

    assert output.shape == torch.Size([3, 1, 4])
    assert attn.shape == torch.Size([3, 2, 1, 5])
    assert torch.allclose(attn.sum(dim=-1), torch.ones_like(attn.sum(dim=-1)))
    assert torch.equal((attn > 0).sum(dim=-1), torch.full((3, 2, 1), 2))

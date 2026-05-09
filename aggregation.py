"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(hidden_dim,)`` or
        ``(k * hidden_dim,)`` if multiple layers are concatenated.

    Student task:
        Replace or extend the skeleton below with alternative layer selection,
        token pooling (mean, max, weighted), or multi-layer fusion strategies.
    """
    real_positions = attention_mask.nonzero(as_tuple=False).flatten()
    if real_positions.numel() == 0:
        real_positions = torch.arange(hidden_states.size(1), device=attention_mask.device)

    last_pos = int(real_positions[-1].item())
    valid_hidden = hidden_states[:, real_positions, :].float()

    n_layers = hidden_states.size(0)
    layer_ids = sorted({max(0, n_layers - 1 - step) for step in (0, 2, 4, 8)})

    window = min(48, valid_hidden.size(1))
    tail_hidden = valid_hidden[:, -window:, :]

    pieces: list[torch.Tensor] = []
    stat_pieces: list[torch.Tensor] = []

    for layer_id in layer_ids:
        layer_all = valid_hidden[layer_id]
        layer_tail = tail_hidden[layer_id]

        last_vec = hidden_states[layer_id, last_pos, :].float()
        mean_all = layer_all.mean(dim=0)
        mean_tail = layer_tail.mean(dim=0)
        max_tail = layer_tail.max(dim=0).values

        pieces.extend(
            [
                last_vec,
                mean_tail,
                mean_all,
                last_vec - mean_tail,
                max_tail - mean_tail,
            ]
        )

        norm_last = last_vec.norm().view(1)
        norm_tail = mean_tail.norm().view(1)
        tail_std = layer_tail.std(dim=0, unbiased=False).mean().view(1)
        cosine = F.cosine_similarity(
            last_vec.view(1, -1), mean_tail.view(1, -1), dim=1
        )
        stat_pieces.append(torch.cat([norm_last, norm_tail, tail_std, cosine]))

    for left, right in zip(layer_ids, layer_ids[1:]):
        prev_last = hidden_states[left, last_pos, :].float()
        next_last = hidden_states[right, last_pos, :].float()
        drift = (next_last - prev_last).norm().view(1)
        cos = F.cosine_similarity(prev_last.view(1, -1), next_last.view(1, -1), dim=1)
        stat_pieces.append(torch.cat([drift, cos]))

    length_feature = torch.tensor(
        [float(real_positions.numel()) / float(hidden_states.size(1))],
        dtype=torch.float32,
        device=hidden_states.device,
    )

    return torch.cat(pieces + stat_pieces + [length_feature], dim=0)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in ``solution.ipynb``.  The
    returned tensor is concatenated with the output of ``aggregate``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(n_geometric_features,)``.  The length
        must be the same for every sample.

    Student task:
        Replace the stub below.  Possible features: layer-wise activation
        norms, inter-layer cosine similarity (representation drift), or
        sequence length.
    """
    real_positions = attention_mask.nonzero(as_tuple=False).flatten()
    if real_positions.numel() == 0:
        return torch.zeros(0, dtype=hidden_states.dtype, device=hidden_states.device)

    last_pos = int(real_positions[-1].item())
    valid_hidden = hidden_states[:, real_positions, :].float()
    last_by_layer = hidden_states[:, last_pos, :].float()

    norms = last_by_layer.norm(dim=1)
    layer_diffs = last_by_layer[1:] - last_by_layer[:-1]
    drift = layer_diffs.norm(dim=1)
    tail = valid_hidden[:, -min(48, valid_hidden.size(1)) :, :]
    tail_var = tail.var(dim=1, unbiased=False).mean(dim=1)

    return torch.cat([norms, drift, tail_var], dim=0)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.ipynb`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.ipynb``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = hidden_dim`` (or larger for multi-layer or geometric
        concatenations).
    """
    agg_features = aggregate(hidden_states, attention_mask)  # (feature_dim,)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features

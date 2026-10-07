"""CPU tests for the ROTATE core (rotate.pursuit). Run with: pytest tests/"""

import pytest
import torch

from rotate.pursuit import (
    BatchedOrthonormalRotations,
    PursuitConfig,
    find_channels,
    kurtosis,
    mask_channel_tokens,
    skewness,
    tail_mask,
)

D, V = 32, 2000
BLOCK_A, BLOCK_B = set(range(0, 40)), set(range(100, 130))


@pytest.fixture(scope="module")
def planted():
    """A vocabulary with two sparse 'concepts' (token blocks sharing orthogonal directions)
    and three neurons that mix them in different proportions."""
    g = torch.Generator().manual_seed(0)
    U = torch.randn(D, V, generator=g)
    q, _ = torch.linalg.qr(torch.randn(D, 2, generator=g))
    dir_a, dir_b = q[:, :1], q[:, 1:2]
    U[:, 0:40] += 6 * dir_a
    U[:, 100:130] += 6 * dir_b
    mix = torch.tensor([[1.0, 1.0], [1.0, 0.6], [0.6, 1.0]])
    neurons = mix @ torch.cat([dir_a.T, dir_b.T]) + 0.02 * torch.randn(3, D, generator=g)
    return U, neurons


def fast_cfg(**kw):
    base = dict(num_channels=2, max_steps=600, learning_rate=1e-2, warmup_steps=20, checkpoint_every=50,
                convergence_min_steps=200, convergence_window=50, convergence_std=1e-4, seed=0)
    base.update(kw)
    return PursuitConfig(**base)


def test_kurtosis_and_skewness_match_definitions():
    x = torch.randn(200_000, generator=torch.Generator().manual_seed(1))
    assert abs(float(kurtosis(x))) < 0.1          # excess kurtosis of a Gaussian
    assert abs(float(skewness(x))) < 0.05
    spiky = torch.zeros(1000)
    spiky[0] = 1.0
    assert float(kurtosis(spiky)) > 900
    assert float(skewness(spiky)) > 30


def test_householder_rotation_preserves_norm():
    rot = BatchedOrthonormalRotations(num_neurons=5, hidden_size=D, k=2, init_scale=1.0)
    x = torch.randn(5, D)
    torch.testing.assert_close(rot(x).norm(dim=1), x.norm(dim=1))


def test_tail_mask_follows_skewness_sign():
    logits = torch.randn(V, generator=torch.Generator().manual_seed(2))
    logits[:10] = 20.0                             # strong positive tail
    mask = tail_mask(logits, std_threshold=4.0, small_skew_eps=0.05)
    assert mask[:10].all() and mask.sum() == 10
    both = tail_mask(-logits, std_threshold=4.0, small_skew_eps=1e9)   # eps large -> both tails
    assert both[:10].all()


def test_find_channels_shapes_and_determinism(planted):
    U, neurons = planted
    a = find_channels(neurons, U, cfg=fast_cfg(), progress=False)
    b = find_channels(neurons, U, cfg=fast_cfg(), progress=False)
    assert a["channels"].shape == (3, 2, D)
    assert a["kurtosis"].shape == a["masked_tokens"].shape == (3, 2)
    assert a["token_weights"].shape == (3, V)
    torch.testing.assert_close(a["channels"], b["channels"], rtol=0, atol=0)

    single = find_channels(neurons[0], U, cfg=fast_cfg(), progress=False)
    assert single["channels"].shape == (2, D)


def _block_of(channel, U, k=25):
    """Which planted block dominates a channel's top-k vocabulary projection (None if neither)."""
    top = set(torch.topk((channel @ U).abs(), k).indices.tolist())
    for name, block in (("A", BLOCK_A), ("B", BLOCK_B)):
        if len(top & block) >= 0.8 * k:
            return name
    return None


def test_channels_recover_planted_concepts(planted):
    U, neurons = planted
    out = find_channels(neurons, U, cfg=fast_cfg(), progress=False)
    neuron_kurt = kurtosis(neurons @ U, axis=1)
    found = set()
    for n in range(neurons.shape[0]):
        # the first channel is sparser than the neuron and monosemantic: one planted block
        assert out["kurtosis"][n, 0] > neuron_kurt[n]
        first = _block_of(out["channels"][n, 0], U)
        assert first is not None
        found.add(first)
        found.update(b for b in (_block_of(out["channels"][n, 1], U),) if b)
        # channels stay close to the neuron (rotation + faithfulness term)
        cos = torch.nn.functional.cosine_similarity(out["channels"][n], neurons[n].unsqueeze(0), dim=1)
        assert (cos > 0.3).all()
    assert found == {"A", "B"}


def test_masking_down_weights_only_the_channel_tail(planted):
    U, _ = planted
    channel = U[:, sorted(BLOCK_A)].mean(dim=1).unsqueeze(0)  # a direction aligned with block A
    weights, counts = mask_channel_tokens(channel, U, torch.ones(1, V), fast_cfg())
    masked = weights[0] < 1
    assert counts[0] == masked.sum() > 0
    masked_ids = set(torch.nonzero(masked).view(-1).tolist())
    assert masked_ids <= BLOCK_A                       # only the channel's own concept is masked
    assert len(masked_ids) >= 0.5 * len(BLOCK_A)
    assert torch.all(weights[0][masked] == fast_cfg().masked_token_weight)

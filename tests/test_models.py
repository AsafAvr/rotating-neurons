"""CPU tests for reading neurons and the vocabulary projection straight from a checkpoint.

Tiny random Gemma-2 and Llama checkpoints are saved with ``transformers``; the tensors returned
by :mod:`rotate.models` must equal the TransformerLens conventions applied to the HF modules
(W_gate = gate_proj.weight.T, W_out = down_proj.weight.T, W_U = lm_head.weight.T, W_E = embed).
"""

import pytest
import torch

transformers = pytest.importorskip("transformers")

from rotate import models  # noqa: E402
from rotate.weights import Checkpoint  # noqa: E402

D_MODEL, D_MLP, VOCAB, LAYERS = 16, 40, 300, 3


def _save(tmp_path_factory, name, config, shard):
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(config).float().eval()
    path = tmp_path_factory.mktemp(name)
    model.save_pretrained(path, max_shard_size="20KB" if shard else "1GB")
    return model, str(path)


@pytest.fixture(scope="module")
def gemma(tmp_path_factory):
    config = transformers.Gemma2Config(
        vocab_size=VOCAB, hidden_size=D_MODEL, intermediate_size=D_MLP, num_hidden_layers=LAYERS,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, tie_word_embeddings=True,
    )
    return _save(tmp_path_factory, "gemma", config, shard=True)


@pytest.fixture(scope="module")
def llama(tmp_path_factory):
    config = transformers.LlamaConfig(
        vocab_size=VOCAB, hidden_size=D_MODEL, intermediate_size=D_MLP, num_hidden_layers=LAYERS,
        num_attention_heads=2, num_key_value_heads=1, tie_word_embeddings=False,
    )
    return _save(tmp_path_factory, "llama", config, shard=False)


@pytest.mark.parametrize("fixture", ["gemma", "llama"])
def test_neurons_match_transformer_lens_conventions(fixture, request):
    hf, path = request.getfixturevalue(fixture)
    mlp = hf.model.layers[1].mlp
    expected = {
        "gate": mlp.gate_proj.weight.T,  # TransformerLens W_gate, neurons are columns
        "in": mlp.up_proj.weight.T,      # W_in
        "out": mlp.down_proj.weight.T,   # W_out, neurons are rows
    }
    idx = [0, 7, D_MLP - 1]
    for mlp_type, W in expected.items():
        got = models.get_neurons(fixture, 1, idx, mlp_type, source=path)
        want = W[:, idx].T if mlp_type != "out" else W[idx]
        torch.testing.assert_close(got, want.detach(), rtol=0, atol=0)
        torch.testing.assert_close(models.get_neuron(fixture, 1, 7, mlp_type, source=path), want[1].detach())


def test_projection_gemma_uses_tied_unembedding(gemma):
    hf, path = gemma
    assert "lm_head.weight" not in Checkpoint(path)  # tied weights are saved once
    U = models.projection_matrix("gemma", 2, source=path)
    torch.testing.assert_close(U, hf.lm_head.weight.T.detach(), rtol=0, atol=0)


def test_projection_llama_switches_to_embedding_in_early_layers(llama):
    hf, path = llama
    late = models.projection_matrix("llama", 13, source=path)
    early = models.projection_matrix("llama", 12, source=path)
    torch.testing.assert_close(late, hf.lm_head.weight.T.detach(), rtol=0, atol=0)
    torch.testing.assert_close(early, hf.model.embed_tokens.weight.T.detach(), rtol=0, atol=0)
    assert not torch.equal(late, early)


def test_checkpoint_rows_and_dtypes(tmp_path):
    from safetensors.torch import save_file

    t = torch.randn(100, 6).to(torch.bfloat16)
    save_file({"a": t, "b": torch.arange(12, dtype=torch.float32).reshape(3, 4)}, tmp_path / "model.safetensors")
    ckpt = Checkpoint(str(tmp_path))
    assert ckpt.info("a")["shape"] == (100, 6) and ckpt.info("a")["dtype"] == torch.bfloat16
    torch.testing.assert_close(ckpt.get("a"), t, rtol=0, atol=0)
    torch.testing.assert_close(ckpt.get("a", rows=[5, 0, 99]), t[[5, 0, 99]], rtol=0, atol=0)
    torch.testing.assert_close(ckpt.get("a", rows=list(range(80))), t[:80], rtol=0, atol=0)  # bulk path
    torch.testing.assert_close(ckpt.get("b"), torch.arange(12.0).reshape(3, 4))
    with pytest.raises(IndexError):
        ckpt.get("a", rows=[100])
    with pytest.raises(KeyError):
        ckpt.get("missing")


def test_default_token_weights_gemma_zeroes_outliers():
    U = torch.randn(8, 300)
    w = models.default_token_weights("gemma", U)
    outliers = models.load_outlier_token_ids()
    assert len(outliers) == 5295
    small = outliers[outliers < 300]
    assert (w[small] < 1).all() and (w[small] == 0).sum() > 0.9 * len(small)  # high-norm ones get 1e-4
    assert ((w == 1) | (w == 0) | (w == 1e-4)).all()
    w_llama = models.default_token_weights("llama", U)
    assert int((w_llama == 1e-4).sum()) == 3  # top 1% of 300 column norms

import json

import mlx.core as mx
import pytest
from mlx_lm.models import llama


def tiny_llama(vocab=64, hidden=128, layers=2, seed=0, logit_scale=1.0, heads=None,
               kv_heads=None, dtype=None):
    """head_dim is 64 unless ``heads`` is given."""
    heads = heads or hidden // 64
    args = llama.ModelArgs(
        model_type="llama", hidden_size=hidden, num_hidden_layers=layers,
        intermediate_size=2 * hidden, num_attention_heads=heads,
        num_key_value_heads=kv_heads or max(1, heads // 2), rms_norm_eps=1e-5,
        vocab_size=vocab, tie_word_embeddings=True,
    )
    mx.random.seed(seed)
    m = llama.Model(args)
    if logit_scale != 1.0:  # sharpen the (otherwise near-uniform) random model
        m.model.norm.weight = m.model.norm.weight * logit_scale
    if dtype is not None:
        m.set_dtype(dtype)
    mx.eval(m.parameters())
    return m


def write_byte_model_dir(path, model, vocab=256):
    """Save a tiny model as an mlx-lm directory with a byte tokenizer."""
    from mlx.utils import tree_flatten
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from tokenizers.pre_tokenizers import ByteLevel

    path.mkdir(parents=True, exist_ok=True)
    a = model.args
    cfg = {"model_type": "llama", "hidden_size": a.hidden_size,
           "num_hidden_layers": a.num_hidden_layers, "intermediate_size": a.intermediate_size,
           "num_attention_heads": a.num_attention_heads,
           "num_key_value_heads": a.num_key_value_heads, "rms_norm_eps": a.rms_norm_eps,
           "vocab_size": a.vocab_size, "tie_word_embeddings": True}
    (path / "config.json").write_text(json.dumps(cfg))
    mx.save_safetensors(str(path / "model.safetensors"), dict(tree_flatten(model.parameters())))
    alphabet = ByteLevel.alphabet()
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    vocab_map = {chr(c): b for b, c in zip(bs, cs)}
    assert set(vocab_map) == set(alphabet)
    tok = Tokenizer(models.BPE(vocab=vocab_map, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tok.decoder = decoders.ByteLevel()
    (path / "tokenizer.json").write_text(tok.to_str())
    (path / "tokenizer_config.json").write_text(json.dumps(
        {"tokenizer_class": "PreTrainedTokenizerFast"}))
    return path

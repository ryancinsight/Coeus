//! Sequence and container module binding contracts.

use super::super::super::support::run_script;

#[test]
fn test_feedforward_module() {
    run_script(
        r#"
import pycoeus

# FeedForward d_model=4, d_ff=8
ffn = pycoeus.FeedForward(4, 8)
ffn_with_dropout = pycoeus.FeedForward(4, 8, dropout_p=0.5)

x = pycoeus.Tensor([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], [2, 4])
out = ffn.forward(x)
assert out.shape == [2, 4], f"FeedForward output shape: {out.shape}"
dropout_out = ffn_with_dropout.forward(x)
assert dropout_out.shape == [2, 4], f"FeedForward dropout output shape: {dropout_out.shape}"
# Output should be finite and not all the same value
vals = out.data
assert any(abs(v) > 0 for v in vals), "FeedForward output all zero"

# SSOT parity: FeedForward.forward equals linear2(gelu(linear1(x))) when dropout=0.
manual = pycoeus.linear(
    pycoeus.f_gelu(pycoeus.linear(x, ffn.linear1.weight, ffn.linear1.bias)),
    ffn.linear2.weight,
    ffn.linear2.bias,
)
for a, b in zip(out.data, manual.data):
    assert abs(a - b) < 1e-9, f"FeedForward SSOT mismatch: {a} vs {b}"

try:
    _ = pycoeus.FeedForward(4, 8, dropout_p=1.0)
    raise AssertionError("dropout_p=1.0 should raise")
except ValueError:
    pass
"#,
    );
}

#[test]
fn test_lstm_gru_cells() {
    run_script(
        r#"
import pycoeus

# ── LSTMCell ──────────────────────────────────────────────────────────
input_size, hidden_size, batch = 4, 8, 2
cell = pycoeus.LSTMCell(input_size, hidden_size)

x = pycoeus.Tensor([float(i)*0.1 for i in range(batch * input_size)], [batch, input_size])
h = pycoeus.zeros([batch, hidden_size])
c = pycoeus.zeros([batch, hidden_size])

h_new, c_new = cell.step(x, h, c)

assert h_new.shape == [batch, hidden_size], f"LSTM h_new shape: {h_new.shape}"
assert c_new.shape == [batch, hidden_size], f"LSTM c_new shape: {c_new.shape}"
# h_new should be non-zero (tanh of non-zero gates * output gate)
assert any(abs(v) > 1e-6 for v in h_new.data), "LSTM h_new all zero"

# Consecutive steps
h2, c2 = cell.step(x, h_new, c_new)
assert h2.shape == [batch, hidden_size]
# State should change across steps
assert any(abs(a - b) > 1e-6 for a, b in zip(h2.data, h_new.data)), "LSTM: state unchanged"

# Parameters: w_ih + b_ih + w_hh + b_hh = 4 (with bias=True by default)
params = cell.parameters()
assert len(params) == 4, f"LSTM parameter count: {len(params)}"

# ── GRUCell ───────────────────────────────────────────────────────────
gru = pycoeus.GRUCell(input_size, hidden_size)

h_gru = pycoeus.zeros([batch, hidden_size])
h_gru_new = gru.step(x, h_gru)

assert h_gru_new.shape == [batch, hidden_size], f"GRU h_new shape: {h_gru_new.shape}"
# h_new bounded in (-1, 1) since tanh * gating
for v in h_gru_new.data:
    assert abs(v) <= 1.0 + 1e-6, f"GRU output out of tanh range: {v}"

# Multiple steps
h2g = gru.step(x, h_gru_new)
assert h2g.shape == [batch, hidden_size]

params_gru = gru.parameters()
assert len(params_gru) == 4, f"GRU parameter count: {len(params_gru)}"
"#,
    );
}

#[test]
fn test_module_list() {
    run_script(
        r#"
import pycoeus

# ── ModuleList ────────────────────────────────────────────────────────
base = pycoeus.Module()
assert base.parameters() == [], f"base parameters: {base.parameters()}"
assert base.is_training is True
base.eval()
assert base.is_training is False
base.train()
assert base.is_training is True
try:
    base.forward(pycoeus.Tensor([1.0], [1]))
    raise AssertionError("base Module.forward should raise")
except NotImplementedError:
    pass

lin1 = pycoeus.Linear(4, 8)
ln = pycoeus.LayerNorm(8)
lin2 = pycoeus.Linear(8, 4)
layers = pycoeus.ModuleList([lin1, ln, lin2])

assert len(layers) == 3, f"len: {len(layers)}"

# Explicit forward (not auto-chained)
x = pycoeus.Tensor([float(i) for i in range(4)], [1, 4])
out = layers[0].forward(x)
assert out.shape == [1, 8], f"layer[0] output: {out.shape}"
out = layers[1].forward(out)
assert out.shape == [1, 8], f"layer[1] output: {out.shape}"
out = layers[2].forward(out)
assert out.shape == [1, 4], f"layer[2] output: {out.shape}"

# parameters(): collects from all sub-modules
params = layers.parameters()
# Linear(4,8): weight[8,4]+bias[8]=2; LayerNorm(8): weight+bias=2; Linear(8,4): weight[4,8]+bias[4]=2 → 6 total
assert len(params) == 6, f"parameter count: {len(params)}"

# negative indexing
last = layers[-1]
assert hasattr(last, 'forward'), "layers[-1] should have forward"

# setitem
new_lin = pycoeus.Linear(4, 8)
layers[0] = new_lin

# empty ModuleList
empty = pycoeus.ModuleList()
assert len(empty) == 0
empty.append(pycoeus.Linear(2, 2))
assert len(empty) == 1

# index out of range
try:
    _ = layers[10]
    raise AssertionError("out-of-range should raise")
except IndexError:
    pass

# zero_grad runs without error
layers.zero_grad()
"#,
    );
}

#[test]
fn test_embedding_padding_idx() {
    run_script(
        r#"
import pycoeus

emb = pycoeus.Embedding(4, 3, padding_idx=0)
assert emb.padding_idx == 0, f"padding_idx: {emb.padding_idx}"
assert emb.weight.data[:3] == [0.0, 0.0, 0.0], f"padding row: {emb.weight.data[:3]}"

idx = pycoeus.Tensor([0.0, 1.0, 2.0, 0.0], [4])
out = emb.forward(idx)
assert out.shape == [4, 3], f"embedding output shape: {out.shape}"
assert out.data[:3] == [0.0, 0.0, 0.0], f"first padding output: {out.data[:3]}"
assert out.data[9:12] == [0.0, 0.0, 0.0], f"second padding output: {out.data[9:12]}"

no_pad = pycoeus.Embedding(2, 2)
assert no_pad.padding_idx is None, f"no padding_idx: {no_pad.padding_idx}"
"#,
    );
}

#[test]
fn test_sequential_module() {
    run_script(
        r#"
import pycoeus

# ── Sequential chains forward calls ────────────────────────────────────
# Build: Linear(4→8) → LayerNorm(8) as a Sequential
lin = pycoeus.Linear(4, 8)
ln = pycoeus.LayerNorm(8)
model = pycoeus.Sequential([lin, ln])

x = pycoeus.Tensor([float(i) for i in range(4)], [1, 4])
out = model.forward(x)
assert out.shape == [1, 8], f"Sequential output shape: {out.shape}"

# ── Parameters collected from all sub-modules ──────────────────────────
params = model.parameters()
# Linear has weight+bias (2), LayerNorm has weight+bias (2) → total 4
assert len(params) == 4, f"parameter count: {len(params)}"

# ── Empty Sequential is identity ──────────────────────────────────────
empty = pycoeus.Sequential([])
x_small = pycoeus.Tensor([1.0, 2.0, 3.0], [3])
out_id = empty.forward(x_small)
assert out_id.data == x_small.data

# ── __len__ and __getitem__ ────────────────────────────────────────────
assert len(model) == 2
m0 = model[0]  # should be the Linear
assert hasattr(m0, 'forward'), "model[0] should have forward"
m_neg = model[-1]  # last item
assert hasattr(m_neg, 'forward'), "model[-1] should have forward"

try:
    model[5]
    raise AssertionError("out-of-range index should raise")
except IndexError:
    pass

# ── Backward gradient flows through chain ─────────────────────────────
x2 = pycoeus.Tensor([float(i) for i in range(4)], [1, 4], requires_grad=True)
out2 = model.forward(x2)
pycoeus.sum(out2).backward()
assert x2.grad is not None, "Sequential backward should produce gradient on input"
"#,
    );
}

#[test]
fn test_transformer_decoder_layer() {
    run_script(
        r#"
import math
import pycoeus

# Construction: valid dropout_p ranges [0.0, 1.0)
for p in (0.0, 0.1, 0.999):
    pycoeus.TransformerDecoderLayer(d_model=4, d_ff=8, num_heads=2, dropout_p=p)
# defaults: num_heads=8, dropout_p=0.0; use a compatible d_model.
pycoeus.TransformerDecoderLayer(d_model=8, d_ff=16)

# Validation: dropout_p must lie in [0.0, 1.0).
for bad in (1.0, 1.5, -0.1):
    try:
        _ = pycoeus.TransformerDecoderLayer(d_model=4, d_ff=8, dropout_p=bad)
        raise AssertionError(f"dropout_p={bad} should raise")
    except ValueError:
        pass
try:
    _ = pycoeus.TransformerDecoderLayer(d_model=4, d_ff=8)
    raise AssertionError("default num_heads=8 should reject d_model=4")
except ValueError:
    pass

# Forward: shape preservation across supported num_heads.
for h in (1, 2, 4):
    dec = pycoeus.TransformerDecoderLayer(d_model=4, d_ff=8, num_heads=h, dropout_p=0.0)
    batch = 1
    seq_tgt = 3
    seq_src = 5
    tgt = pycoeus.Tensor([0.01 * (i + 1) for i in range(batch * seq_tgt * 4)], [batch, seq_tgt, 4])
    memory = pycoeus.Tensor([0.02 * (i + 1) for i in range(batch * seq_src * 4)], [batch, seq_src, 4])
    out = dec.forward(tgt, memory)
    assert out.shape == [batch, seq_tgt, 4], f"num_heads={h} shape={out.shape}"
    # SSOT parity: decoder forward equals explicit pre-LN composition when dropout=0.
    n1 = pycoeus.layer_norm(tgt, 4, dec.norm1.weight, dec.norm1.bias, eps=1e-5)
    x1 = tgt + dec.self_attn.forward(n1)
    n2 = pycoeus.layer_norm(x1, 4, dec.norm2.weight, dec.norm2.bias, eps=1e-5)
    x2 = x1 + dec.cross_attn.forward_cross(n2, memory, memory)
    n3 = pycoeus.layer_norm(x2, 4, dec.norm3.weight, dec.norm3.bias, eps=1e-5)
    manual = x2 + dec.ffn.forward(n3)
    for a, b in zip(out.data, manual.data):
        assert abs(a - b) < 1e-9, f"decoder layer SSOT mismatch (h={h}): {a} vs {b}"
    # Real PyLayer cannot no-op on these inputs (random init weights produce
    # non-trivial logits); absence of zeros guards against silent fall-through.
    assert any(abs(v) > 1e-6 for v in out.data), f"num_heads={h} all-zero output"

# Stateful wrapper surface
dec = pycoeus.TransformerDecoderLayer(d_model=4, d_ff=8, num_heads=2)
assert len(dec.parameters()) == 26, "stateful decoder layer exposes all learnable parameters"
dec.zero_grad()
"#,
    );
}

#[test]
fn test_transformer_encoder_bindings() {
    run_script(
        r#"
import pycoeus

# Encoder layer construction validates dropout_p and preserves [batch, seq, d_model].
for p in (0.0, 0.2, 0.999):
    pycoeus.TransformerEncoderLayer(d_model=4, d_ff=8, num_heads=2, dropout_p=p)
for bad in (1.0, -0.1):
    try:
        _ = pycoeus.TransformerEncoderLayer(d_model=4, d_ff=8, num_heads=2, dropout_p=bad)
        raise AssertionError(f"encoder layer dropout_p={bad} should raise")
    except ValueError:
        pass

src = pycoeus.Tensor([0.01 * (i + 1) for i in range(1 * 3 * 4)], [1, 3, 4])
for h in (1, 2, 4):
    enc_layer = pycoeus.TransformerEncoderLayer(d_model=4, d_ff=8, num_heads=h, dropout_p=0.0)
    out = enc_layer.forward(src)
    assert out.shape == [1, 3, 4], f"encoder layer h={h} shape={out.shape}"
    assert any(abs(v) > 1e-6 for v in out.data), f"encoder layer h={h} all-zero output"
    # SSOT parity: forward == pre-LN composition when dropout=0.
    n1 = pycoeus.layer_norm(src, 4, enc_layer.norm1.weight, enc_layer.norm1.bias, eps=1e-5)
    x1 = src + enc_layer.self_attn.forward(n1)
    n2 = pycoeus.layer_norm(x1, 4, enc_layer.norm2.weight, enc_layer.norm2.bias, eps=1e-5)
    manual = x1 + enc_layer.ffn.forward(n2)
    for a, b in zip(out.data, manual.data):
        assert abs(a - b) < 1e-9, f"encoder layer SSOT mismatch (h={h}): {a} vs {b}"

try:
    bad = pycoeus.TransformerEncoderLayer(d_model=4, d_ff=8, num_heads=3, dropout_p=0.0)
    bad.forward(src)
    raise AssertionError("unsupported encoder layer num_heads should raise")
except ValueError:
    pass

# Encoder stack construction validates dropout_p and supported const-generic pairs.
for p in (0.0, 0.25):
    pycoeus.TransformerEncoder(d_model=4, d_ff=8, num_heads=2, num_layers=2, dropout_p=p)
for bad in (1.0, -0.1):
    try:
        _ = pycoeus.TransformerEncoder(d_model=4, d_ff=8, num_heads=2, num_layers=2, dropout_p=bad)
        raise AssertionError(f"encoder dropout_p={bad} should raise")
    except ValueError:
        pass

for h, n in ((1, 1), (2, 2), (4, 1)):
    enc = pycoeus.TransformerEncoder(d_model=4, d_ff=8, num_heads=h, num_layers=n, dropout_p=0.0)
    out = enc.forward(src)
    assert out.shape == [1, 3, 4], f"encoder h={h} n={n} shape={out.shape}"
    assert any(abs(v) > 1e-6 for v in out.data), f"encoder h={h} n={n} all-zero output"

try:
    bad = pycoeus.TransformerEncoder(d_model=4, d_ff=8, num_heads=2, num_layers=3, dropout_p=0.0)
    bad.forward(src)
    raise AssertionError("unsupported encoder layer count should raise")
except ValueError:
    pass

# Sinusoidal encoding validates even d_model and adds position signal.
for bad_d in (0, 3):
    try:
        _ = pycoeus.SinusoidalEncoding(max_len=8, d_model=bad_d)
        raise AssertionError(f"sinusoidal d_model={bad_d} should raise")
    except ValueError:
        pass

pe = pycoeus.SinusoidalEncoding(max_len=8, d_model=4)
zeros = pycoeus.Tensor([0.0] * (1 * 3 * 4), [1, 3, 4])
pos = pe.forward(zeros)
assert pos.shape == [1, 3, 4], f"sinusoidal shape={pos.shape}"
assert any(abs(v) > 1e-6 for v in pos.data), "sinusoidal output should contain position signal"
"#,
    );
}

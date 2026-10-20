"""Smoke tests on random data: shapes, masks, filling and the objective."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recast import (RecAST, causal_fill, hide_blocks, hide_point, original_validity,  # noqa: E402
                    recast_loss, window_masks)


def _inputs(b=2, n=7, p=12, seed=0):
    rng = np.random.RandomState(seed)
    adj = (rng.rand(n, n) > 0.6).astype(np.float32)
    values = rng.randn(b, p, n).astype(np.float32)
    observed = rng.rand(b, p, n) > 0.1
    retained = rng.rand(b, p, n) > 0.3
    m, h = window_masks(observed, retained)
    filled = causal_fill(values, m)
    tod = np.broadcast_to((np.arange(p) / 288.0)[None, :, None], (b, p, n))
    dow = np.full((b, p, n), 3.0)
    x = torch.tensor(np.stack([filled, tod, dow], axis=-1), dtype=torch.float32)
    return adj, x, torch.tensor(m[..., None]), torch.tensor(h[..., None]), torch.tensor(values[..., None])


def test_shapes_and_loss():
    torch.manual_seed(0)
    adj, x, m, h, values = _inputs()
    model = RecAST(num_nodes=7, adj=adj)
    forecast, recon = model(x, m, return_reconstruction=True)
    assert forecast.shape == (2, 12, 7, 1) and recon.shape == (2, 12, 7, 1)
    target = torch.randn_like(forecast)
    valid = torch.ones_like(target, dtype=torch.bool)
    loss, parts = recast_loss(forecast, target, valid, recon, values, h, mean=50.0, std=10.0)
    loss.backward()
    assert torch.isfinite(loss) and parts['reconstruction'] > 0


def test_reconstruction_head_does_not_change_forecast():
    adj, x, m, _, _ = _inputs()
    torch.manual_seed(1)
    with_head = RecAST(num_nodes=7, adj=adj).eval()
    without = RecAST(num_nodes=7, adj=adj, reconstruction=False).eval()
    state = {k: v for k, v in with_head.state_dict().items() if not k.startswith('recon_proj')}
    without.load_state_dict(state)
    assert torch.equal(with_head(x, m), without(x, m))


def test_causal_fill():
    values = np.array([[[1.0], [2.0], [3.0], [4.0]]])
    available = np.array([[[False], [True], [False], [True]]])
    filled = causal_fill(values, available)
    assert filled.ravel().tolist() == [0.0, 2.0, 2.0, 4.0]


def test_hiding_budget():
    rng = np.random.RandomState(0)
    raw = rng.rand(500, 20) * 60
    raw[rng.rand(500, 20) < 0.05] = 0
    observed = original_validity(raw)
    for retained in (hide_point(observed, 0.2, np.random.default_rng(0)),
                     hide_blocks(observed, 0.2, np.random.default_rng(0))):
        hidden = observed & ~retained
        assert abs(hidden.sum() / observed.sum() - 0.2) < 0.02


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print(name, 'ok')

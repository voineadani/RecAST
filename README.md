# RecAST

RecAST (Reconstruction-Assisted Spatio-Temporal transformer) is a traffic forecasting model for
sensor networks with missing input readings. This repository contains the model, its training
objective and the mask utilities used to build its inputs, as described in:

> G.-D. Voinea, M. Duguleană, R. G. Boboc and F. Gîrbacia, "Observation-Aware Traffic Forecasting
> Under Missing Sensor Data: A Controlled Evaluation of Masking, Reconstruction, and Imputation,"
> submitted to *IEEE Access*.

## Method

- **Inputs.** Each sensor's 12-step history (60 min at 5-min resolution) is standardized with the
  mean and SD of all valid training readings. Unavailable entries are forward-filled from the
  same sensor's most recent available value in the window; entries with no earlier value are set
  to the training mean. The availability mask `M` is passed to the model with the filled values,
  so a filled value is never presented as an observation.
- **Embedding.** A linear projection of the filled value (24), time-of-day and day-of-week
  embeddings (24 each), a node embedding (80) and a linear embedding of `M` (8), giving D = 160.
- **Blocks.** Three blocks of temporal self-attention followed by a mask-conditioned spatial layer.
  The spatial layer computes a graph-convolution update (normalized binarized adjacency plus a
  learned adaptive adjacency) and a spatial-attention update. A two-output sigmoid gate, which sees
  the representation, both updates, `M` and the graph-weighted availability `Â M`, weights the two
  updates.
- **Output.** One linear layer maps the 12 step representations of each sensor to 12 forecasts.
- **Objective.** `L = MAE_V(ŷ, y) + λ · MAE_H(x̂, x)` with λ = 0.5, both terms in source units.
  `V` marks valid forecast targets and `H` marks valid input readings that were hidden
  synthetically during training. The reconstruction head is used only in training.

## Masks

| Symbol | Meaning |
|---|---|
| `O` | original validity: raw reading finite and strictly positive |
| `S` | synthetic retention: False where a valid reading is hidden |
| `M = O & S` | availability, given to the model |
| `H = O & ~S` | hidden readings, the only reconstruction targets |

Training uses 16 corrupted copies of the training timeline, each with point loss or single-sensor
block loss (10–30 min gaps) at 10, 20 or 40% of valid readings. Epoch `e` uses copy `e mod 16`, and
each training window is shown in its corrupted form with probability 0.5.

## Installation

```
pip install -r requirements.txt
```

Tested with Python 3.8 and PyTorch 2.2.

## Usage

```python
import numpy as np
import torch
from recast import RecAST, causal_fill, window_masks, recast_loss

# adj: (N, N) sensor adjacency; values: (B, 12, N) standardized readings
# observed (O) and retained (S): (B, 12, N) Boolean arrays
M, H = window_masks(observed, retained)
filled = causal_fill(values, M)
x = torch.tensor(np.stack([filled, time_of_day, day_of_week], axis=-1), dtype=torch.float32)
mask = torch.tensor(M[..., None])

model = RecAST(num_nodes=adj.shape[0], adj=adj)
forecast, reconstruction = model(x, mask, return_reconstruction=True)
loss, terms = recast_loss(forecast, y, y_valid, reconstruction,
                          torch.tensor(values[..., None]), torch.tensor(H[..., None]),
                          mean=train_mean, std=train_std, recon_weight=0.5)
loss.backward()
```

At inference, call `model(x, mask)`. The forecast is in standardized units; multiply by the
training SD and add the training mean to obtain source units.

Default hyperparameters and the per-dataset optimization settings used in the paper are listed in
`configs/recast.json`. Data loading is not included; METR-LA, PEMS-BAY and PEMSD8 are public
benchmark datasets.

## Tests

```
python tests/test_recast.py
```

## License

See `LICENSE`.

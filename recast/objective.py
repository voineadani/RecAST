"""Training objective of RecAST.

L = MAE_V(y_hat, y) + lambda * MAE_H(x_hat, x)

Both terms are computed in source units after inverse standardization. Each term
averages over its own valid entries: V marks originally valid forecast targets and
H marks originally valid input readings that were hidden synthetically. The
reconstruction target is always the original measurement, never a filled value.
The auxiliary term is zero when H is empty.
"""
import torch


def masked_mae(prediction, target, valid, allow_empty=False):
    """Mean absolute error over entries where `valid` (bool) is True."""
    if prediction.shape != target.shape or valid.shape != target.shape or valid.dtype != torch.bool:
        raise ValueError('prediction, target and a Boolean validity mask must have the same shape')
    if not valid.any():
        if allow_empty:
            return prediction.sum() * 0
        raise ValueError('No valid entries: the loss is undefined')
    return (prediction[valid] - target[valid]).abs().mean()


def recast_loss(forecast, target, target_valid, reconstruction, reconstruction_target,
                hidden, mean, std, recon_weight=0.5):
    """Forecast MAE plus weighted reconstruction MAE, both in source units.

    Args:
        forecast, target, target_valid: (B, Q, N, 1); standardized values and V.
        reconstruction, reconstruction_target, hidden: (B, P, N, 1); the
            reconstruction head output, the original standardized input values and H.
        mean, std: standardization parameters fitted on valid training readings.
        recon_weight: lambda (0.5 in the paper).
    Returns:
        (total loss, dict with the two unweighted terms).
    """
    def inverse(z):
        return z.float() * std + mean

    forecast_term = masked_mae(inverse(forecast), inverse(target), target_valid)
    if recon_weight:
        recon_term = masked_mae(inverse(reconstruction), inverse(reconstruction_target),
                                hidden, allow_empty=True)
    else:
        recon_term = forecast_term.new_zeros(())
    total = forecast_term + recon_weight * recon_term
    return total, {'forecast': forecast_term.detach(), 'reconstruction': recon_term.detach()}

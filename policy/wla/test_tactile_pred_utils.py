"""Regression tests for tactile prediction helper shapes."""

import torch

from policy.wla.tactile_pred_utils import invert_target_projection


def test_invert_target_projection_shape():
    target_proj = torch.nn.Linear(1024, 64)
    pred_latent = torch.randn(32, 64)

    concat_latent = invert_target_projection(target_proj, pred_latent)

    assert concat_latent.shape == (32, 1024)


if __name__ == "__main__":
    test_invert_target_projection_shape()
    print("tactile_pred_utils_test: OK")

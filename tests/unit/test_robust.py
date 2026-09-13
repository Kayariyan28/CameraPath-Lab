"""An internal OpenCV assertion on one degenerate frame must not abort a shot (I12)."""

import cv2
import numpy as np

from app.tracking.robust import failure_counts, robust


def test_estimator_error_becomes_no_model():
    def exploding(*_a, **_k):
        raise cv2.error("OpenCV(4.14.0) estimator.cpp:353: error: (-215:Assertion failed) !model.empty()")
    exploding.__name__ = "fakeEstimator"
    assert robust(exploding, np.zeros((4, 2))) == (None, None)
    assert failure_counts().get("fakeEstimator", 0) >= 1


def test_successful_call_passes_through():
    src = np.random.default_rng(0).uniform(0, 100, (30, 2)).astype(np.float32)
    model, mask = robust(cv2.estimateAffinePartial2D, src, src + 3.0)
    assert model is not None and np.allclose(model[:, 2], 3.0, atol=1e-3)


def test_non_opencv_errors_are_not_swallowed():
    """Only estimator failures map to 'no model'; a programming error must surface."""
    import pytest
    def buggy(*_a, **_k):
        raise TypeError("wrong arguments")
    with pytest.raises(TypeError):
        robust(buggy)

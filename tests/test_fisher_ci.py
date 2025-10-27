from pykt.metrics_utils import fisher_z_ci


def test_insufficient_samples():
    low, high = fisher_z_ci(0.5, 3)
    assert low is None and high is None


def test_zero_correlation():
    low, high = fisher_z_ci(0.0, 10)
    assert low is not None and high is not None and low < 0.01 and high > -0.01


def test_positive_correlation_mid():
    low, high = fisher_z_ci(0.5, 30)
    assert low is not None and high is not None and low < 0.5 < high


def test_near_one_correlation():
    low, high = fisher_z_ci(0.999999, 50)
    assert low is None and high is None


def test_negative_correlation():
    low, high = fisher_z_ci(-0.4, 25)
    assert low is not None and high is not None and low < -0.4 < high

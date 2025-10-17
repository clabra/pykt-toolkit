import torch
from pykt.models.gainakt2_exp import GainAKT2Exp


def _make_dummy(batch_size=2, seq_len=5, num_c=10, device='cpu'):
    # random but deterministic
    torch.manual_seed(123)
    q = torch.randint(0, num_c, (batch_size, seq_len), device=device)
    r = torch.randint(0, 2, (batch_size, seq_len), device=device)
    return q, r


def test_interpretability_loss_all_zero_weights():
    num_c = 10
    model = GainAKT2Exp(
        num_c=num_c,
        seq_len=20,
        non_negative_loss_weight=0.0,
        monotonicity_loss_weight=0.0,
        mastery_performance_loss_weight=0.0,
        gain_performance_loss_weight=0.0,
        sparsity_loss_weight=0.0,
        consistency_loss_weight=0.0,
    )
    q, r = _make_dummy(num_c=num_c)
    out = model.forward_with_states(q, r)
    loss = out['interpretability_loss']
    assert isinstance(loss, float) or (isinstance(loss, torch.Tensor) and loss.item() == 0.0)
    # When all weights zero we expect a Python float 0.0 (implementation returns scalar float) or a 0 tensor
    assert float(loss) == 0.0


def test_interpretability_loss_nonzero_components():
    num_c = 10
    model = GainAKT2Exp(
        num_c=num_c,
        seq_len=20,
        non_negative_loss_weight=0.0,  # still architecturally enforced; keep zero weight
        monotonicity_loss_weight=0.1,
        mastery_performance_loss_weight=0.2,
        gain_performance_loss_weight=0.2,
        sparsity_loss_weight=0.05,
        consistency_loss_weight=0.1,
    )
    q, r = _make_dummy(num_c=num_c)
    out = model.forward_with_states(q, r)
    loss = out['interpretability_loss']
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor when any weight > 0"
    assert loss.item() > 0.0, "Expected positive interpretability loss with non-zero weights"


def test_consistency_loss_effect():
    # Compare with and without consistency weight holding other weights at zero to isolate effect
    num_c = 10
    q, r = _make_dummy(num_c=num_c)
    base_model = GainAKT2Exp(
        num_c=num_c,
        seq_len=20,
        non_negative_loss_weight=0.0,
        monotonicity_loss_weight=0.0,
        mastery_performance_loss_weight=0.0,
        gain_performance_loss_weight=0.0,
        sparsity_loss_weight=0.0,
        consistency_loss_weight=0.0,
    )
    cons_model = GainAKT2Exp(
        num_c=num_c,
        seq_len=20,
        non_negative_loss_weight=0.0,
        monotonicity_loss_weight=0.0,
        mastery_performance_loss_weight=0.0,
        gain_performance_loss_weight=0.0,
        sparsity_loss_weight=0.0,
        consistency_loss_weight=0.5,
    )
    base_loss = base_model.forward_with_states(q, r)['interpretability_loss']
    cons_loss = cons_model.forward_with_states(q, r)['interpretability_loss']
    assert float(base_loss) == 0.0
    assert cons_loss.item() > 0.0, "Consistency loss should contribute when weight > 0"


def test_compute_interpretability_loss_direct():
    """Directly exercise compute_interpretability_loss to distinguish zero vs non-zero weight behavior."""
    num_c = 8
    q, r = _make_dummy(num_c=num_c, seq_len=6)

    # Model with all zero weights
    zero_model = GainAKT2Exp(
        num_c=num_c,
        seq_len=20,
        non_negative_loss_weight=0.0,
        monotonicity_loss_weight=0.0,
        mastery_performance_loss_weight=0.0,
        gain_performance_loss_weight=0.0,
        sparsity_loss_weight=0.0,
        consistency_loss_weight=0.0,
    )
    out_zero = zero_model.forward_with_states(q, r)
    loss_zero = zero_model.compute_interpretability_loss(
        out_zero['projected_mastery'],
        out_zero['projected_gains'],
        out_zero['predictions'],
        q,
        r,
    )
    # With all weights zero, loss should be a float (implementation starts at 0.0) or zero tensor
    assert float(loss_zero) == 0.0

    # Model with selective non-zero weights
    weighted_model = GainAKT2Exp(
        num_c=num_c,
        seq_len=20,
        non_negative_loss_weight=0.0,
        monotonicity_loss_weight=0.05,
        mastery_performance_loss_weight=0.1,
        gain_performance_loss_weight=0.1,
        sparsity_loss_weight=0.02,
        consistency_loss_weight=0.05,
    )
    out_w = weighted_model.forward_with_states(q, r)
    loss_w = weighted_model.compute_interpretability_loss(
        out_w['projected_mastery'],
        out_w['projected_gains'],
        out_w['predictions'],
        q,
        r,
    )
    assert isinstance(loss_w, torch.Tensor), "Expected tensor loss when any weight > 0"
    assert loss_w.item() > 0.0, "Non-zero weights should yield positive interpretability loss"


if __name__ == '__main__':
    test_interpretability_loss_all_zero_weights()
    test_interpretability_loss_nonzero_components()
    test_consistency_loss_effect()
    print("Interpretability loss tests passed.")

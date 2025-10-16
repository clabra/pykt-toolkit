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


if __name__ == '__main__':
    test_interpretability_loss_all_zero_weights()
    test_interpretability_loss_nonzero_components()
    test_consistency_loss_effect()
    print("Interpretability loss tests passed.")

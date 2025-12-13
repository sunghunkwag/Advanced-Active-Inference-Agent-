import pytest

torch = pytest.importorskip("torch")

from transition_model import TransitionModel
from tri_neuro_model import TriNeuroHybrid


def test_basic_transition_shape():
    latent_dim = 16
    num_actions = 4
    context_dim = 8
    hidden_dim = 32
    model = TransitionModel(latent_dim, num_actions, context_dim, hidden_dim)
    batch_size = 10
    z = torch.randn(batch_size, latent_dim)
    action = torch.randint(0, num_actions, (batch_size,), dtype=torch.long)
    context = torch.randn(batch_size, context_dim)
    out = model(z, action, context)
    assert out.shape == (batch_size, latent_dim), f"Unexpected output shape {out.shape}"


def test_tri_neuro_output_shape_and_reset():
    latent_dim = 16
    num_actions = 3
    context_dim = 5
    hidden_dim = 32
    model = TransitionModel(
        latent_dim, num_actions, context_dim,
        hidden_dim=hidden_dim,
        use_tri_neuro=True,
        tri_hidden_dim=16,
        tri_ema_decay=0.9,
    )
    batch_size = 7
    z = torch.randn(batch_size, latent_dim)
    actions = torch.randint(0, num_actions, (batch_size,), dtype=torch.long)
    context = torch.randn(batch_size, context_dim)
    out1 = model(z, actions, context)
    assert out1.shape == (batch_size, latent_dim)
    assert model.tri_neuro.global_state is not None
    out2 = model(z * 0.5, actions, context * 0.3)
    assert out2.shape == (batch_size, latent_dim)
    assert not torch.allclose(model.tri_neuro.global_state, out1), "Global state did not update"
    model.reset_states()
    assert model.tri_neuro.global_state is None
    assert model.tri_neuro._gru_state is None


def test_gating_weights_sum_to_one():
    input_dim = 12
    context_dim = 6
    latent_dim = 8
    hidden_dim = 16
    model = TriNeuroHybrid(input_dim, context_dim, latent_dim, hidden_dim)
    batch_size = 5
    x = torch.randn(batch_size, input_dim)
    context = torch.randn(batch_size, context_dim)
    with torch.no_grad():
        gates = torch.softmax(model.gating_net(context), dim=-1)
        sums = gates.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums)), f"Gates do not sum to 1: {sums}"

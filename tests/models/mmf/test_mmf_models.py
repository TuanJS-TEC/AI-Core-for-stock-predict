import pytest, torch
from ai_core.models.mmf import config as C
from ai_core.models.mmf.models import FusionNet

def inputs(n=4):
    return torch.randn(n, C.LP, 25), torch.randn(n, C.LT, 14), torch.randn(n, 10), torch.zeros(n, dtype=torch.long)

@pytest.mark.parametrize("kind", C.PRICE_KINDS)
@pytest.mark.parametrize("arm", list(C.ARMS))
def test_every_arm_and_kind_gives_three_logits(arm, kind):
    br, fu = C.ARMS[arm]
    assert FusionNet(br, fu, 15, 25, 14, 10, kind)(*inputs()).shape == (4, 3)

def test_absent_branch_is_never_read():
    br, fu = C.ARMS["no_text"]
    net = FusionNet(br, fu, 15, 25, 14, 10).eval()
    p, t, f, k = inputs()
    assert torch.equal(net(p, t, f, k), net(p, t * 0 + 99, f, k))

def test_attention_weights_sum_to_one():
    br, fu = C.ARMS["fusion_attention"]
    net = FusionNet(br, fu, 15, 25, 14, 10).eval()
    net(*inputs())
    assert net.last_weights.shape == (4, 3)
    assert torch.allclose(net.last_weights.sum(1), torch.ones(4))

def test_last_step_reaches_the_output():
    net = FusionNet(("price",), "none", 15, 25, 14, 10).eval()
    p, t, f, k = inputs()
    q = p.clone(); q[:, -1] += 5
    assert not torch.equal(net(p, t, f, k), net(q, t, f, k))

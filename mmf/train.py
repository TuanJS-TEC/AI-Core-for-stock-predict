"""Training loop: AdamW, cross-entropy, early stopping on tune NLL, best-epoch weights returned."""
import copy

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C


def _tensors(s, device):
    return [torch.as_tensor(a).to(device) for a in (s.x_price, s.x_text, s.x_fund, s.ticker)]


def predict_logits(net, s, device, batch=1024):
    """Eval-mode logits (N, 3) as a CPU numpy array."""
    net.eval()
    x = _tensors(s, device)
    with torch.no_grad():
        return np.concatenate([net(*(a[i:i + batch] for a in x)).cpu().numpy() for i in range(0, len(s), batch)])


def _labelled(s):
    keep = s.y >= 0
    return type(s)(*(getattr(s, f)[keep] if getattr(s, f) is not None else None for f in s.__dataclass_fields__))


def fit_net(make_net, train, tune, seed, device, max_epochs=C.MAX_EPOCHS, patience=C.PATIENCE):
    """Train on labelled rows; return {net (best epoch, eval, on device), best_epoch, tune_nll}."""
    train, tune = _labelled(train), _labelled(tune)
    torch.manual_seed(seed)
    net = make_net().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=C.LR, weight_decay=C.WD)
    x, y = _tensors(train, device), torch.as_tensor(train.y).to(device)
    y_tune = torch.as_tensor(tune.y)
    gen = torch.Generator().manual_seed(seed)
    best, best_epoch, best_state = np.inf, -1, None
    for epoch in range(max_epochs):
        net.train()
        for idx in torch.randperm(len(y), generator=gen).split(C.BATCH):
            idx = idx.to(device)
            opt.zero_grad()
            F.cross_entropy(net(*(a[idx] for a in x)), y[idx]).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        nll = F.cross_entropy(torch.as_tensor(predict_logits(net, tune, device)), y_tune).item()
        if nll < best:
            best, best_epoch, best_state = nll, epoch, copy.deepcopy(net.state_dict())
        elif epoch - best_epoch >= patience:
            break
    net.load_state_dict(best_state)
    return {"net": net.eval(), "best_epoch": best_epoch, "tune_nll": best}

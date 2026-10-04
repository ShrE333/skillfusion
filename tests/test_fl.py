import numpy as np
from skillfusion.fl.aggregate import ClientUpdate, fedavg, fuse, skill_code, split_params, l2_drift

def test_fedavg_weighted():
    a = ClientUpdate("a", 1, {"trunk.w": np.array([0.0, 0.0])})
    b = ClientUpdate("b", 3, {"trunk.w": np.array([4.0, 8.0])})
    assert np.allclose(fedavg([a, b])["trunk.w"], [3.0, 6.0])

def test_fedavg_rejects_mismatch():
    for ups in ([ClientUpdate("a", 1, {"trunk.w": np.zeros(1)}), ClientUpdate("b", 1, {"trunk.v": np.zeros(1)})],
                [ClientUpdate("a", 0, {"trunk.w": np.zeros(1)})]):
        try: fedavg(ups)
        except ValueError: continue
        raise AssertionError

def test_skill_codes_orthonormal_and_stable():
    C = np.stack([skill_code(i) for i in range(16)])
    assert np.allclose(C @ C.T, np.eye(16), atol=1e-5)
    assert np.allclose(skill_code(3), skill_code(3))

def test_fuse_keeps_private_and_rejects_dupes():
    u1 = ClientUpdate("a", 1, {"trunk.w": np.ones(2)}, {"head.a.w": np.ones(1)})
    u2 = ClientUpdate("b", 1, {"trunk.w": np.ones(2)}, {"head.b.w": np.zeros(1)})
    f = fuse(fedavg([u1, u2]), [u1, u2])
    assert set(f) == {"trunk.w", "head.a.w", "head.b.w"}
    try: fuse({}, [u1, ClientUpdate("c", 1, {}, {"head.a.w": np.ones(1)})])
    except ValueError: pass
    else: raise AssertionError

def test_federated_training_fuses_three_skills():
    """3 clients, shared linear trunk T (true), private heads H_s: y = H_s T x. FedAvg on trunk only."""
    rng = np.random.default_rng(0)
    d, h, o = 6, 4, 2
    T_true = rng.standard_normal((h, d))
    H_true = {s: rng.standard_normal((o, h)) for s in "abc"}
    data = {}
    for s in "abc":
        X = rng.standard_normal((200, d)); data[s] = (X, X @ T_true.T @ H_true[s].T)
    T = rng.standard_normal((h, d)) * 0.1
    H = {s: rng.standard_normal((o, h)) * 0.1 for s in "abc"}
    def loss(s, T, Hs):
        X, Y = data[s]; return float(((X @ T.T @ Hs.T - Y) ** 2).mean())
    start = np.mean([loss(s, T, H[s]) for s in "abc"])
    for rnd in range(60):
        ups = []
        for s in "abc":
            Tl = T.copy(); Hs = H[s]; X, Y = data[s]
            for _ in range(20):
                Z = X @ Tl.T; E = Z @ Hs.T - Y
                gH = 2 * E.T @ Z / len(X); gT = 2 * (E @ Hs).T @ X / len(X)
                Tl -= 0.02 * gT; Hs -= 0.02 * gH
            H[s] = Hs
            ups.append(ClientUpdate(s, len(X), {"trunk.T": Tl}, {f"head.{s}.H": Hs}))
        T = fedavg(ups)["trunk.T"]
    end = np.mean([loss(s, T, H[s]) for s in "abc"])
    assert end < 0.05 * start, (start, end)

def test_drift():
    assert l2_drift({"trunk.w": np.zeros(3)}, {"trunk.w": np.ones(3)}) == np.sqrt(3)

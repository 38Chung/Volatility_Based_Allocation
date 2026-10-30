"""選 λ 規則的檢查：λ = c × λ_max、並列取中間點、驗證段簡化 S1 扣成本後 Sharpe、重估後必有非零係數。"""
import numpy as np
import pytest
from sklearn.linear_model import Lasso

import config as C
import model as MD


def test_c_grid_excludes_one():
    assert len(MD.C_GRID) == 29 and MD.C_GRID.max() < 1 and np.isclose(MD.C_GRID.min(), 1e-3)


def test_middle_of_ties():
    assert MD.middle_of_ties([7]) == 7
    assert MD.middle_of_ties([5, 3, 4]) == 4
    assert MD.middle_of_ties([3, 4]) == 3                 # 偶數個取中間偏小
    assert MD.middle_of_ties([2, 3, 4, 5]) == 3


def test_lambda_max_threshold():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3)); y = X @ np.array([0.5, 0.0, 0.2]) + rng.normal(size=400)
    lm = MD.lambda_max(X, y)
    fit = lambda a: Lasso(alpha=a, positive=True, fit_intercept=False, max_iter=10000).fit(X, y).coef_
    assert not np.any(fit(lm * 1.001) > 0)                # λ ≥ λ_max → 全為 0
    assert np.any(fit(lm * 0.95) > 0)                     # λ < λ_max → 至少一個非零（太貼近 λ_max 時 sklearn 的收斂檢查會在第一步前就停）
    assert np.any(fit(lm * MD.C_GRID.max()) > 0)


def test_sim_net_sharpe_hand_calc():
    # 3 檔、K = 1、緩衝 0：第 0、1 週選 A，第 2 週換成 B（換手 Σ|Δw| = 2）；第 2 週 A 排第 2 名
    scores = np.array([[3, 2, 1], [3, 2, 1], [2, 3, 1]], float)
    y = np.array([[0.01, 0.00, -0.01], [0.02, 0.01, 0.00], [0.00, 0.03, -0.02]])
    el = np.ones_like(y, bool)
    got = MD.sim_net_sharpe(scores, y, el, K=1, buffer=0, cost_bps=15)
    net = np.array([0.01, 0.02, 0.03 - 15e-4 * 2])
    assert np.isclose(got, net.mean() / net.std(ddof=1))
    # 排名緩衝：A 掉到第 2 名但在前 K + 1 內 → 續抱，不換手
    got_b = MD.sim_net_sharpe(scores, y, el, K=1, buffer=1, cost_bps=15)
    net_b = np.array([0.01, 0.02, 0.00])
    assert np.isclose(got_b, net_b.mean() / net_b.std(ddof=1))
    # 被剔出股票池（eligible False）的持股一定賣出，即使排名還在緩衝內
    el2 = el.copy(); el2[2, 0] = False
    got_c = MD.sim_net_sharpe(scores, y, el2, K=1, buffer=1, cost_bps=15)
    assert np.isclose(got_c, net.mean() / net.std(ddof=1))


@pytest.fixture(scope="module")
def ctx():
    import run_stage2 as R2
    panel, sti = R2.load_truncated()
    c = R2.Context(panel, sti, verbose=False)
    c.fit_models(which=("lasso260",))
    return c


def test_refit_nonzero_and_logged(ctx):
    lg = ctx.model_logs["lasso260"]
    for col in ["lmax_front", "lmax_full", "c", "n_tied", "n_nonzero", "val_net_sharpe", "val_ic_at_c"]:
        assert col in lg.columns, col
    ok = lg["lmax_full"] > 0
    assert (lg.loc[ok, "n_nonzero"] > 0).all()                                 # c < 1 → 重估必有非零係數
    assert (lg["all_zero"] == ~ok).all()                                       # 全為 0 只能來自 λ_max ≤ 0
    assert np.allclose(lg.loc[ok, "alpha"], lg.loc[ok, "c"] * lg.loc[ok, "lmax_full"])
    assert set(np.round(lg["c"].dropna(), 12)) <= set(np.round(MD.C_GRID, 12))
    print("\nλ_max ≤ 0 的期數：", int((~ok).sum()), "；c 的分布：", lg["c"].round(4).value_counts().head(5).to_dict())


def test_lambda_validation_inside_window(ctx):
    lg = ctx.model_logs["lasso260"]
    for _, r in lg.sample(20, random_state=C.SEED).iterrows():
        assert r["val_last_y_end"] <= np.datetime64(r["signal"])               # 驗證段的報酬在訊號日前就結束
        assert r["val_last_idx"] - r["val_first_idx"] + 1 == C.LAMBDA_VALID_WEEKS

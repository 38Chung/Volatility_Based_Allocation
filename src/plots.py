"""報告用的靜態圖（PNG）。顏色固定對應組合（不因圖而改），色盤檢查過對比與色差：
B0 是灰色參考線；B0′ 藍、B1 橘、A1 青、A2 黃、S1 洋紅、S2 綠、S3 紫。
B0′（藍）和 S3（紫）色差不足，不放在同一張圖。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

import config as C

COLORS = {"B0": "#8a8984", "B0'": "#2a78d6", "B1": "#eb6834", "A1": "#1baf7a", "A2": "#eda100",
          "S1": "#e87ba4", "S2": "#008300", "S3": "#6250d6"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
VALID_SHADE = "#f0efea"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _shade_valid(ax, label=True):
    a, b = pd.Timestamp(C.VALID_START), pd.Timestamp(C.VALID_END)
    ax.axvspan(a, b, color=VALID_SHADE, zorder=0, linewidth=0)
    if label:
        ax.text(a + (b - a) / 2, 1.0, "Validation", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=8, color=INK2)


def _end_labels(ax, series, fmt):
    """線尾直接標籤（文字用墨色，旁邊的小色塊表示身分）；自動錯開避免重疊。"""
    items = sorted(((s.dropna().iloc[-1], name) for name, s in series.items()), reverse=True)
    lo, hi = ax.get_ylim()
    log = ax.get_yscale() == "log"
    tr = (lambda v: np.log(v)) if log else (lambda v: v)
    inv = (lambda v: np.exp(v)) if log else (lambda v: v)
    gap = (tr(hi) - tr(lo)) * 0.045
    placed = []
    for v, name in items:
        y = tr(v)
        for p in placed:
            if abs(y - p) < gap:
                y = p - gap
        placed.append(y)
        x = series[name].dropna().index[-1]
        ax.plot([x], [inv(y)], marker="s", markersize=6, color=COLORS[name], clip_on=False, linestyle="none")
        ax.annotate(f"{name}  {fmt(v)}", xy=(x, inv(y)), xytext=(9, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK, annotation_clip=False)


def _font():
    for f in ["PingFang TC", "Heiti TC", "Arial Unicode MS", "Noto Sans CJK TC"]:
        if f in {x.name for x in matplotlib.font_manager.fontManager.ttflist}:
            plt.rcParams["font.sans-serif"] = [f] + plt.rcParams["font.sans-serif"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def nav_chart(nav, path, note, title="Cumulative NAV (net of costs)", shade_valid=True, log=True, k_label="K"):
    """兩個面板：全部股票的組合 / 因子選股的組合，都以 B0（STI）為灰色參考。"""
    _font()
    panels = [("All tradable constituents", ["B0'", "B1", "A1", "A2"]), (f"Top {k_label} by factor score", ["S1", "S2", "S3"])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True, facecolor=SURFACE)
    for ax, (ptitle, names) in zip(axes, panels):
        _style(ax)
        if shade_valid:
            _shade_valid(ax)
        ax.plot(nav.index, nav["B0"], color=COLORS["B0"], linewidth=1.6, linestyle=(0, (4, 2)))
        for n in names:
            ax.plot(nav.index, nav[n], color=COLORS[n], linewidth=1.6)
        if log:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(mticker.FixedLocator([0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0]))
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_locator(mticker.NullLocator())
        else:
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.3f}"))
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m-%d"))
        ax.set_title(ptitle, loc="left", fontsize=11, color=INK, pad=14)
        _end_labels(ax, {n: nav[n] for n in ["B0"] + names}, (lambda v: f"{v:.2f}") if log else (lambda v: f"{v - 1:+.1%}"))
    axes[0].set_ylabel("NAV (start = 1" + (", log scale)" if log else ")"), color=INK2, fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=INK)
    fig.text(0.01, 0.005, note, fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 0.94, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def freq_chart(weekly, monthly, b0, path, note):
    """月頻對照實驗：S3 週頻（實線）vs 月頻（虛線），B0 灰色參考；三條線從同一天起點 = 1。"""
    _font()
    start = max(weekly.dropna().index[0], monthly.dropna().index[0])
    lines = {"S3 weekly": weekly, "S3 monthly": monthly, "B0": b0}
    lines = {k: v.loc[start:] / v.loc[start:].iloc[0] for k, v in lines.items()}
    style = {"S3 weekly": dict(color=COLORS["S3"], linewidth=1.8),
             "S3 monthly": dict(color=COLORS["S3"], linewidth=1.8, linestyle=(0, (5, 2)), alpha=0.75),
             "B0": dict(color=COLORS["B0"], linewidth=1.4, linestyle=(0, (4, 2)))}
    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor=SURFACE)
    _style(ax)
    _shade_valid(ax)
    for k, s in lines.items():
        ax.plot(s.index, s, **style[k])
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mticker.FixedLocator([0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0]))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    lo, hi = np.log(ax.get_ylim())
    gap, placed = (hi - lo) * 0.05, []
    for v, k in sorted(((s.iloc[-1], k) for k, s in lines.items()), reverse=True):   # 線尾標籤，太近就往下錯開
        y = np.log(v)
        for p in placed:
            if abs(y - p) < gap:
                y = p - gap
        placed.append(y)
        ax.annotate(f"{k}  {v:.2f}", xy=(lines[k].index[-1], np.exp(y)), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK, annotation_clip=False)
    ax.set_ylabel(f"NAV ({start.date()} = 1, log scale)", color=INK2, fontsize=9)
    fig.suptitle("S3 weekly vs monthly rebalancing: cumulative NAV (net of costs)", x=0.01, ha="left", fontsize=13, color=INK)
    fig.text(0.01, 0.005, note, fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 0.88, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def drawdown_chart(nav, path, note, names=("B0", "A2", "S2", "S3")):
    """小倍數：每個組合一格，共用 y 軸。"""
    _font()
    fig, axes = plt.subplots(len(names), 1, figsize=(11, 1.9 * len(names)), sharex=True, sharey=True,
                             facecolor=SURFACE)
    for ax, n in zip(axes, names):
        _style(ax)
        _shade_valid(ax, label=(n == names[0]))
        dd = nav[n] / nav[n].cummax() - 1
        ax.fill_between(dd.index, dd.values, 0, color=COLORS[n], alpha=0.25, linewidth=0)
        ax.plot(dd.index, dd.values, color=COLORS[n], linewidth=1.5)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
        i = dd.idxmin()
        ax.annotate(f"Max drawdown {dd.min():.1%} ({i:%Y-%m})", xy=(i, dd.min()), xytext=(6, -2),
                    textcoords="offset points", fontsize=8.5, color=INK, va="top")
        ax.text(0.003, 0.08, n, transform=ax.transAxes, fontsize=10, color=INK, fontweight="bold")
    axes[0].set_ylim(min(-0.45, axes[0].get_ylim()[0]), 0.02)
    fig.suptitle("Drawdown from previous peak (net of costs)", x=0.01, ha="left", fontsize=13, color=INK)
    fig.text(0.01, 0.005, note, fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def cost_chart(perf, path, note, order=("B0'", "B1", "A1", "A2", "S1", "S2", "S3")):
    """啞鈴圖：每組的年化報酬，扣成本前（空心灰）→ 扣成本後（實心墨色），訓練期 / 驗證期兩格。"""
    _font()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), facecolor=SURFACE)
    for ax, per, title in zip(axes, ["train", "valid"], ["Training (2014-01 – 2022-12)", "Validation (2023-01 – 2026-07)"]):
        _style(ax)
        ax.grid(axis="y", visible=False); ax.grid(axis="x", color=GRID, linewidth=0.8)
        d = perf[perf.period == per].pivot(index="portfolio", columns="cost", values="ann_return").reindex(list(order))
        y = np.arange(len(d))[::-1]
        for yi, (n, row) in zip(y, d.iterrows()):
            ax.plot([row["gross"], row["net"]], [yi, yi], color=GRID, linewidth=3, solid_capstyle="round", zorder=1)
            ax.scatter([row["gross"]], [yi], s=70, facecolor=SURFACE, edgecolor=INK2, linewidth=1.6, zorder=2)
            ax.scatter([row["net"]], [yi], s=70, color=INK, zorder=3)
            drag = row["gross"] - row["net"]
            if drag > 0.003:
                ax.text(max(row["gross"], row["net"]), yi + 0.28, f"Cost −{drag:.1%}", fontsize=8, color=INK2,
                        ha="right")
        ax.set_yticks(y); ax.set_yticklabels(d.index, fontsize=10, color=INK)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlim(min(0, d.values.min() - 0.01), d.values.max() + 0.015)
    axes[0].scatter([], [], s=70, facecolor=SURFACE, edgecolor=INK2, linewidth=1.6, label="Before costs")
    axes[0].scatter([], [], s=70, color=INK, label="After costs (15 bps one-way)")
    fig.legend(loc="upper right", frameon=False, fontsize=9, ncol=2, bbox_to_anchor=(0.99, 0.99))
    fig.suptitle("Annualised return: before vs after costs", x=0.01, ha="left", fontsize=13, color=INK)
    fig.text(0.01, 0.005, note, fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def gamma_chart(vx, path, note):
    """E2：驗證期 Sharpe、波動、最大回撤隨傾斜強度 γ 變化（S2、A1）。"""
    _font()
    g2 = vx[(vx.series == "E2") & (vx.period == "valid")].copy()
    g2["gamma"] = g2["option"].str.replace("gamma", "").astype(float)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), facecolor=SURFACE)
    for ax, col, lab, pct in zip(axes, ["sharpe", "ann_vol", "max_drawdown"], ["Sharpe", "Annualised volatility", "Max drawdown"],
                                 [False, True, True]):
        _style(ax)
        for pname, mk in [("S2", "o"), ("A1", "s")]:
            d = g2[g2.portfolio == pname].sort_values("gamma")
            ax.plot(d["gamma"], d[col], color=COLORS[pname], linewidth=2, marker=mk, markersize=7)
            ax.annotate(pname, xy=(d["gamma"].iloc[-1], d[col].iloc[-1]), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=9, color=INK)
        if pct:
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=1))
        ax.set_xticks([0, 0.5, 1, 2]); ax.set_xlabel("γ (w ∝ σ̂^−γ; 0 = equal weight, 1 = baseline)", fontsize=9, color=INK2)
        ax.set_title(lab, loc="left", fontsize=11, color=INK)
        ax.set_xlim(-0.15, 2.35)
    fig.suptitle("Effect of tilt strength γ (validation, net of costs)", x=0.01, ha="left", fontsize=13, color=INK)
    fig.text(0.01, 0.005, note, fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def exposure_chart(expo, path, note, test_start=None):
    """A2、S3 每週的目標曝險（股票部位）。"""
    _font()
    fig, ax = plt.subplots(figsize=(12, 3.6), facecolor=SURFACE)
    _style(ax)
    _shade_valid(ax)
    if test_start is not None:
        ax.axvline(pd.Timestamp(test_start), color=INK2, linewidth=1, linestyle=(0, (2, 2)))
        ax.text(pd.Timestamp(test_start), 1.0, " Test", transform=ax.get_xaxis_transform(), fontsize=8,
                color=INK2, va="bottom")
    for n in ["A2", "S3"]:
        ax.step(expo.index, expo[n], where="post", color=COLORS[n], linewidth=1.6)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylim(min(0.4, expo.min().min() - 0.05), 1.05)
    _end_labels(ax, {n: expo[n] for n in ["A2", "S3"]}, lambda v: f"{v:.0%}")
    fig.suptitle("Total exposure (equity share of NAV; the rest is cash earning no interest)", x=0.01, ha="left", fontsize=13, color=INK)
    fig.text(0.01, 0.005, note, fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 0.95, 0.95))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def lasso_heatmap(coef, path, note):
    """Lasso 係數（因子 × 訊號日），單色藍色色階；0 = 背景色。因子依入選頻率排序。"""
    _font()
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    order = (coef > 0).mean().sort_values(ascending=False).index
    c = coef[order]
    fig, ax = plt.subplots(figsize=(13, 6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    x = matplotlib.dates.date2num(c.index.to_pydatetime())
    im = ax.imshow(c.T.values, aspect="auto", cmap=cmap, interpolation="nearest", vmin=0,
                   extent=[x[0], x[-1], len(order) - 0.5, -0.5])
    ax.xaxis_date()
    freq = (c > 0).mean()
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{f}  {freq[f]:.0%}" for f in order], fontsize=9, color=INK)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors=INK2, length=0)
    _shade_line = pd.Timestamp(C.VALID_START)
    ax.axvline(matplotlib.dates.date2num(_shade_line), color=INK2, linewidth=1, linestyle=(0, (2, 2)))
    ax.text(matplotlib.dates.date2num(_shade_line), -0.9, " Validation start", fontsize=8, color=INK2, va="bottom")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=8, colors=INK2, length=0)
    cb.set_label("Coefficient β (≥ 0)", fontsize=9, color=INK2)
    fig.suptitle("Rolling Lasso coefficients (re-estimated weekly; % after each factor = selection rate)", x=0.01, ha="left", fontsize=13, color=INK)
    fig.text(0.01, 0.005, note, fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


SECTOR_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def sector_chart(secw, path, note, title):
    """產業權重（股票部位內，加總 1）堆疊面積；顏色依類別固定順序，堆疊之間留背景色細縫。"""
    _font()
    rel = secw.div(secw.sum(axis=1), axis=0).fillna(0)
    fig, ax = plt.subplots(figsize=(12, 4.2), facecolor=SURFACE)
    _style(ax)
    ax.stackplot(rel.index, rel.T.values, colors=SECTOR_COLORS[:rel.shape[1]], linewidth=0, labels=rel.columns)
    ax.set_ylim(0, 1); ax.set_xlim(rel.index[0], rel.index[-1])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=9)   # 跟堆疊上下順序一致
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=INK)
    fig.text(0.01, 0.005, note, fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def ic_decay_heatmap(decay, path, note):
    """IC 衰減（訓練期）：因子 × 持有週數，雙色發散色階（紅 < 0 < 藍，中間灰），格內標數值。"""
    _font()
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    ic = decay["IC_mean"].unstack()
    t = decay["NW_t"].unstack()
    ic = ic.reindex(ic[1].sort_values(ascending=False).index)
    t = t.reindex(ic.index)
    lim = np.nanmax(np.abs(ic.values))
    fig, ax = plt.subplots(figsize=(7.5, 7.5), facecolor=SURFACE)
    ax.imshow(ic.values, cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for i in range(ic.shape[0]):
        for j in range(ic.shape[1]):
            v, tv = ic.iat[i, j], t.iat[i, j]
            ax.text(j, i, f"{v:+.3f}" + ("*" if abs(tv) >= 1.96 else ""), ha="center", va="center", fontsize=8.5,
                    color=INK)
    ax.set_xticks(range(ic.shape[1])); ax.set_xticklabels([f"{h}w" for h in ic.columns], fontsize=9, color=INK)
    ax.set_yticks(range(ic.shape[0])); ax.set_yticklabels(ic.index, fontsize=9, color=INK)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, ic.shape[1]), minor=True); ax.set_yticks(np.arange(-0.5, ic.shape[0]), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2); ax.tick_params(which="minor", length=0)
    fig.suptitle("IC decay (training-period weekly Rank IC, sign-adjusted; * = |NW t| ≥ 1.96)", x=0.01, ha="left", fontsize=12,
                 color=INK)
    fig.text(0.01, 0.005, note, fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)

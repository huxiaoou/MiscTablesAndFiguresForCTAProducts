import os
import numpy as np
import pandas as pd
from skyrim.riften import CNAV
from winterhold2 import CPlotLines, CPlotSingleNavWithDrawdown


def get_corr(df, x: str, y: str, w: int):
    x_aver = df[x].rolling(window=w).mean()
    y_aver = df[y].rolling(window=w).mean()
    xy_aver = (df[x] * df[y]).rolling(window=w).mean()
    xx_aver = (df[x] * df[x]).rolling(window=w).mean()
    yy_aver = (df[y] * df[y]).rolling(window=w).mean()
    cov_xy = xy_aver - x_aver * y_aver
    cov_xx = xx_aver - x_aver * x_aver
    cov_yy = yy_aver - y_aver * y_aver
    return cov_xy / np.sqrt(cov_xx * cov_yy)


def get_rolling_corr(df: pd.DataFrame, win: int = 21) -> pd.DataFrame:
    res = {}
    for v0 in df.columns:
        for v1 in df.columns:
            if v0 < v1:
                res[f"{v0}-{v1}"] = get_corr(df, v0, v1, win)
    return pd.DataFrame(res)


pd.set_option("float_format", "{:.6f}".format)
pd.set_option("display.width", 0)

input_dir = os.path.join("..", "data", "input")
output_dir = os.path.join("..", "data", "output")

sub_return_files = {
    "qian": "qian.return.20230630.xlsx",
    "yuex": "yuex.return.20230630.xlsx",
    "huxo": "huxo.return.20230630.xlsx",
    # "qian": "qian.return.xlsx",
    # "yuex": "yuex.return.xlsx",
    # "huxo": "huxo.return.xlsx",
}

bgn_date, stp_date = "2018-01-01", "2023-07-01"
performance_indicators = ["hold_period_return", "annual_return", "annual_volatility",
                          "sharpe_ratio", "calmar_ratio",
                          "max_drawdown_scale", "max_drawdown_scale_idx",
                          "q01", "q05"]

return_data = {}
for sub_return_id, sub_return_file in sub_return_files.items():
    sub_return_df = pd.read_excel(
        os.path.join(input_dir, sub_return_file), dtype={"trade_date": str}).set_index("trade_date")
    return_data[sub_return_id] = sub_return_df["return"]

raw_return_df = pd.DataFrame(return_data).fillna(0)
raw_return_df.index = raw_return_df.index.map(lambda z: z[0:4] + "-" + z[4:6] + "-" + z[6:8])
filter_dates = (raw_return_df.index >= bgn_date) & (raw_return_df.index < stp_date)
net_return_df = raw_return_df.loc[filter_dates].copy()
net_nav_df = (net_return_df + 1).cumprod()
net_nav_df_since_2023 = net_nav_df.loc[net_nav_df.index >= "2023-01-01"]
net_nav_df_since_2023 = net_nav_df_since_2023 / net_nav_df_since_2023.iloc[0]

# --- adjust return
net_return_std = net_return_df.std()
weight_srs = 1 / net_return_std
weight_srs = weight_srs / weight_srs.sum()
adj_return_df: pd.DataFrame = net_return_df.multiply(weight_srs, axis=1)
adj_return_df["GH"] = adj_return_df.sum(axis=1)
adj_return_df.to_csv(os.path.join(output_dir, "adj_return.csv"), float_format="%.6f")
adj_return_df[["GH"]].to_csv(os.path.join(output_dir, "adj_return_gh.csv"), float_format="%.6f")
q = adj_return_df.quantile((0, 0.01, 0.02, 0.05, 1))

# --- adjust nav
adj_nav_df = (adj_return_df + 1).cumprod()
adj_nav_df_since_2022 = adj_nav_df.loc[adj_return_df.index >= "2022-01-01"]
adj_nav_df_since_2023 = adj_nav_df.loc[adj_return_df.index >= "2023-01-01"]

# --- performance summary
summary_data = {}
for sub_return_id in net_return_df.columns:
    nav = CNAV(net_return_df[sub_return_id], t_annual_rf_rate=0, t_type="ret")
    nav.cal_all_indicators()
    summary_data[sub_return_id] = nav.to_dict(t_type="eng")
nav = CNAV(adj_return_df["GH"], t_annual_rf_rate=0, t_type="ret")
nav.cal_all_indicators(t_qs=(1, 5))
summary_data["comb"] = nav.to_dict(t_type="eng")
summary_df = pd.DataFrame.from_dict(summary_data, orient="index")
summary_df = summary_df[performance_indicators]
summary_df.to_csv(
    os.path.join(output_dir, "performance_strategy.csv"),
    index_label="strategy", float_format="%.6f",
)

# --- by year
summary_by_year_data = {}
adj_return_df["trade_year"] = adj_return_df.index.map(lambda z: z[0:4])
for trade_year, trade_year_df in adj_return_df.groupby(by="trade_year"):
    nav = CNAV(trade_year_df["GH"], t_annual_rf_rate=0, t_type="ret")
    nav.cal_all_indicators(t_qs=(1, 5))
    summary_by_year_data[trade_year] = nav.to_dict(t_type="eng")
summary_by_year_df = pd.DataFrame.from_dict(summary_by_year_data, orient="index")
summary_by_year_df = summary_by_year_df[performance_indicators]
summary_by_year_df.to_csv(
    os.path.join(output_dir, "performance_strategy_GH_by_year.csv"),
    index_label="trade_year", float_format="%.6f",
)
adj_return_df.drop(axis=1, labels="trade_year", inplace=True)

# --- rolling corr
renamed_adj_ret_df = adj_return_df[list(sub_return_files)].rename(mapper={
    "qian": "子策略一",
    "yuex": "子策略二",
    "huxo": "子策略三",
}, axis=1)
adj_ret_rolling_cor = get_rolling_corr(renamed_adj_ret_df, win=63)

print("=" * 120)
print("调整前日收益率")
print(net_return_df)

print("=" * 120)
print("调整后日收益率")
print(adj_return_df)

print("=" * 120)
print("累计净值")
print(adj_nav_df)

print("=" * 120)
print("日波动率")
print(adj_return_df.std())

print("=" * 120)
print("相关性")
print(adj_return_df.corr())

print("=" * 120)
print("绩效摘要")
print(summary_df)

print("=" * 120)
print("滚动相关性")
print(adj_ret_rolling_cor)

artist = CPlotLines(
    plot_df=net_nav_df[["qian", "yuex", "huxo"]].rename(mapper={
        "qian": "子策略一",
        "yuex": "子策略二",
        "huxo": "子策略三",
    }, axis=1),
    fig_name="comb_nav_sub", fig_save_dir=output_dir, fig_save_type="PNG",
    fig_size=(16, 4),
    xtick_count=9, xtick_label_rotation=0,
    style="seaborn-v0_8-poster",
    line_color=['#000080', '#4169E1', '#B0C4DE'],
    xtick_label_size=16, ytick_label_size=16,
    legend_fontsize=16,
)
artist.plot()

artist = CPlotLines(
    plot_df=adj_ret_rolling_cor,
    fig_name="adj_ret_rolling_corr", fig_save_dir=output_dir, fig_save_type="PNG",
    fig_size=(16, 7),
    xtick_count=9, xtick_label_rotation=0,
    style="seaborn-v0_8-poster",
    line_color=['#000080', '#4169E1', '#B0C4DE'],
    xtick_label_size=16, ytick_label_size=16,
    legend_fontsize=16, legend_loc="lower center",
)
artist.plot()

artist = CPlotSingleNavWithDrawdown(
    nav_srs=adj_nav_df["GH"],
    nav_label="国海量化", drawdown_label="回撤幅度",
    fig_name="GH", fig_save_dir=output_dir, fig_save_type="PNG", style="seaborn-v0_8-poster",
    fig_size=(18, 17),
    xtick_count=8, xtick_label_rotation=0,
    nav_line_color=['#000080'], drawdown_color=["#4169E1"], drawdown_alpha=0.5,
    xtick_label_size=20, ytick_label_size=20, ytick_label_size_twin=20,
    legend_fontsize=20,
)
artist.plot()

print("=" * 120)
print("组合策略按年绩效摘要")
print(summary_by_year_df)

print("=" * 120)
print("单日最大亏损")
print(q)

print("=" * 120)

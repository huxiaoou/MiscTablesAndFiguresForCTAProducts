import os
import numpy as np
import pandas as pd
from winterhold2 import CPlotLines, CPlotLinesTwinxBar


def cal_annual_factor(xs: pd.Series, n_lbl: str, d_lbl: str):
    n_mon = int(xs[n_lbl].split(".")[0][-2:])
    d_mon = int(xs[d_lbl].split(".")[0][-2:])
    d = d_mon - n_mon
    if d < 0:
        d += 12
    return 12 / d


fee = 1e-4

pd.set_option("float_format", "{:.6f}".format)
pd.set_option("display.width", 0)

input_dir = os.path.join("..", "data", "input")
output_dir = os.path.join("..", "data", "output")

raw_file = "major_minor.Y.DCE.xlsx"
raw_path = os.path.join(input_dir, raw_file)
raw_df = pd.read_excel(raw_path, sheet_name="major_minor.Y.DCE", dtype={"trade_date": str})
raw_df["af"] = raw_df.apply(cal_annual_factor, args=("n_contract", "d_contract"), axis=1)
raw_df["TS"] = (raw_df["close_n"] / raw_df["close_d"] - 1) * raw_df["af"]
raw_df["sign"] = np.sign(raw_df["TS"].shift(1).fillna(0))
raw_df["p_ret"] = raw_df["index_ret"] * raw_df["sign"]
raw_df["Y.DCE"] = (raw_df["index_ret"] / 100 + 1).cumprod()
raw_df["Strategy"] = ((raw_df["p_ret"] / 100 - fee) + 1).cumprod()
raw_df["trade_date"] = raw_df["trade_date"].map(lambda z: z[0:4] + "-" + z[4:6] + "-" + z[6:8])
raw_df.set_index("trade_date", inplace=True)
print(raw_df)

artist = CPlotLinesTwinxBar(
    plot_df=raw_df[["Y.DCE", "Strategy", "TS"]].rename(mapper={"Y.DCE": "豆油指数", "Strategy": "策略净值", "TS": "TS因子"}, axis=1),
    primary_cols=["豆油指数", "策略净值"], secondary_cols=["TS因子"],
    line_color=["#4169E1", "#8B4513"], line_style=["-.", "-"],
    bar_color=["#B0C4DE"], bar_width=1.0, bar_alpha=0.6,
    xtick_spread=252,
    fig_name="strategy_example", fig_save_type="PNG", fig_save_dir=output_dir,
    style="seaborn-v0_8-poster",
    xtick_count=9, xtick_label_size=16, ytick_label_size=16, ytick_label_size_twin=16,
    fig_size=(27, 6),
)
artist.plot()

for ins in ["Y", "AU"]:
    ts_df = pd.read_excel(raw_path, sheet_name=f"ts-{ins}").set_index("contract")
    print(ts_df)
    artist = CPlotLines(
        plot_df=ts_df, line_width=4,
        fig_name=f"ts_example_{ins}", fig_save_type="PNG", fig_save_dir=output_dir,
        line_color=["#00008B"],
        xtick_label_size=20, ytick_label_size=20,
        fig_size=(17, 3),
    )
    artist.plot()

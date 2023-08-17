import os
import numpy as np
import pandas as pd
from skyrim.winterhold import plot_twinx
from winterhold2 import CPlotLines, CPlotLinesTwinxBar

pd.set_option("float_format", "{:.6f}".format)
pd.set_option("display.width", 0)

input_dir = os.path.join("..", "data", "input")
output_dir = os.path.join("..", "data", "output")

raw_file = "extra.xlsx"
raw_path = os.path.join(input_dir, raw_file)
raw_df = pd.read_excel(raw_path, sheet_name="daily_ret")
raw_df["Date"] = raw_df["Date"].map(lambda _: _.strftime("%Y-%m-%d"))
raw_df.set_index("Date", inplace=True)
raw_df.rename(mapper={"NH0100.NHF": "南华商品", "000905.SH": "中证500"}, axis=1, inplace=True)

volatility_df = raw_df.rolling(window=21).std(ddof=1) * np.sqrt(252)
nav_df = (raw_df / 100 + 1).cumprod()
nav_df = nav_df.loc[nav_df.index >= "2018-01-01"]
nav_df = nav_df / nav_df.iloc[0]

merged_df = pd.merge(left=nav_df, right=volatility_df, left_index=True, right_index=True, how="left", suffixes=("", "-21日滚动波动率"))
merged_df: pd.DataFrame = merged_df.loc[merged_df.index < "2023-07-01"]
merged_df.sort_index(inplace=True)
print(merged_df)
print(merged_df.median())

for mkt_idx in ["南华商品", "中证500"]:
    vol_id = mkt_idx + "-21日滚动波动率"
    artist = CPlotLinesTwinxBar(
        plot_df=merged_df[[mkt_idx, vol_id]], primary_cols=[mkt_idx], second_cols=[vol_id],
        fig_size=(16, 6), xtick_count=10,
        line_color=["#0000CD"],
        bar_color=["#DC143C"], ylim_twin=(0, 100), bar_width=1, bar_alpha=0.6,
        fig_name=f"volatility_mkt_idx-{mkt_idx}", fig_save_type="PNG", fig_save_dir=output_dir,
        style="seaborn-v0_8-poster",
        xtick_label_size=16, ytick_label_size=16,
        legend_fontsize=16,
    )
    artist.plot()

nav_df_since_2021 = nav_df.loc[nav_df.index >= "2021-01-01"]
artist = CPlotLines(
    plot_df=nav_df_since_2021[["南华商品"]],
    fig_name="南华商品_since_2021", fig_save_dir=output_dir, fig_save_type="PNG",
    fig_size=(18, 4),
    xtick_count=10, xtick_label_rotation=0,
    style="seaborn-v0_8-poster",
    line_color=["#000080", "#4169E1", "#B0C4DE"],
    xtick_label_size=18, ytick_label_size=18,
    legend_fontsize=18,
)
artist.plot()

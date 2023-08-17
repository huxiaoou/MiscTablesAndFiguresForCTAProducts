import os
import numpy as np
import pandas as pd
from skyrim.winterhold import plot_twinx
from winterhold2 import CPlotLines

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

merged_df = pd.merge(left=nav_df, right=volatility_df, left_index=True, right_index=True, how="left", suffixes=("", "-21日滚动波动率"))
merged_df: pd.DataFrame = merged_df.loc[merged_df.index < "2023-07-01"]
merged_df.sort_index(inplace=True)
print(merged_df)
print(merged_df.median())

for mkt_idx in ["南华商品", "中证500"]:
    vol_id = mkt_idx + "-21日滚动波动率"
    plot_twinx(
        t_plot_df=merged_df[[mkt_idx, vol_id]],
        t_primary_cols=[mkt_idx], t_secondary_cols=[vol_id],
        t_primary_kind="line", t_secondary_kind="bar",
        t_xtick_span=189,
        t_fig_name=f"volatility_mkt_idx-{mkt_idx}",
        t_fig_size=(16, 6),
        t_save_type="PNG",
        t_save_dir=output_dir,
        t_style="fast",
        t_tick_label_size=16,
        t_tick_label_rotation=0,
        t_secondary_ylim=(0, 100),
        t_primary_colormap=None,
        t_secondary_colormap="gist_gray",
    )

nav_df_since_2021 = nav_df.loc[nav_df.index >= "2021-01-01"]
artist = CPlotLines(
    plot_df=nav_df_since_2021[["南华商品"]],
    fig_name="南华商品_since_2021", fig_save_dir=output_dir, fig_save_type="PNG",
    fig_size=(18, 4),
    xtick_count=10, xtick_label_rotation=0,
    style="seaborn-v0_8-poster",
    line_color=['#000080', '#4169E1', '#B0C4DE'],
    xtick_label_size=18, ytick_label_size=18,
    legend_fontsize=18,
)
artist.plot()
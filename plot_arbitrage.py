import os
import pandas as pd
from winterhold2 import CPlotLines

input_dir = os.path.join("..", "data", "input")
output_dir = os.path.join("..", "data", "output")

bgn_date, end_date = "2012-01-01", "2023-08-31"

src_file = "Y_M_ratio.xlsx"
src_path = os.path.join(input_dir, src_file)
src_df = pd.read_excel(src_path, header=1)
src_df.dropna(axis=0, subset=["价差(收)"], inplace=True)
src_df["trade_date"] = src_df["时间"].map(lambda _: _.strftime("%Y-%m-%d"))
src_df.rename(mapper={"价差(收)": "油粕价格比"}, axis=1, inplace=True)
src_df = src_df[["trade_date", "油粕价格比"]].set_index("trade_date")
src_df = src_df.truncate(before=bgn_date, after=end_date)
print(src_df)

artist = CPlotLines(
    plot_df=src_df,
    fig_name="arbitrage_Y_M", fig_save_dir=output_dir, fig_save_type="PNG",
    fig_size=(26, 6),
    xtick_count=10, xtick_label_rotation=0,
    style="seaborn-v0_8-poster",
    # line_color=['#000080', '#4169E1', '#B0C4DE', '#DC143C'],
    # line_style=['-', '-', '-', '-.'],
    line_color=['#000080', '#DC143C'],
    line_style=['-', '-.'],
    xtick_label_size=16, ytick_label_size=16,
    legend_fontsize=16,
)
artist.plot()

import os
import pandas as pd
from winterhold2 import CPlotBars

input_dir = os.path.join("..", "data", "input")
output_dir = os.path.join("..", "data", "output")

src_pos_file = "model-pes-pes-pes.csv.gz"
src_pos_df = pd.read_csv(os.path.join(output_dir, src_pos_file), dtype={"trade_date": str}).set_index("trade_date")
src_pos_df.rename(mapper={"qian-pes": "子策略一", "yuex-pes": "子策略二", "huxo-pes": "子策略三"}, axis=1, inplace=True)
src_pos_df.index = src_pos_df.index.map(lambda z: "-".join([z[0:4], z[4:6], z[6:8]]))
print(src_pos_df)

# plot dynamic
artist = CPlotBars(
    plot_df=src_pos_df, bar_color=["#00008B", "#4169E1", "#B0C4DE"], stacked=True,
    fig_size=(10, 3), fig_name="weight_sub_strategy_dyn_min_uty", fig_save_dir=output_dir, fig_save_type="PNG",
    xtick_spread=3, xtick_label_rotation=90,
    xtick_label_size=12, ytick_label_size=12,
    legend_fontsize=12, legend_loc=None,
)
artist.plot()

# plot vanilla
src_pos_df["子策略一"] = 1 / 3
src_pos_df["子策略二"] = 1 / 3
src_pos_df["子策略三"] = 1 / 3
artist = CPlotBars(
    plot_df=src_pos_df, bar_color=["#00008B", "#4169E1", "#B0C4DE"], stacked=True,
    fig_size=(10, 3), fig_name="weight_sub_strategy_vanilla", fig_save_dir=output_dir, fig_save_type="PNG",
    xtick_spread=3, xtick_label_rotation=90,
    xtick_label_size=12, ytick_label_size=12,
    legend_fontsize=12, legend_loc="upper left"
)
artist.plot()

# plot static eql vol / data is from the print output of opt2.py
src_pos_df["子策略一"] = 0.20023706
src_pos_df["子策略二"] = 0.40008547
src_pos_df["子策略三"] = 0.39967748
artist = CPlotBars(
    plot_df=src_pos_df, bar_color=["#00008B", "#4169E1", "#B0C4DE"], stacked=True,
    fig_size=(10, 3), fig_name="weight_sub_strategy_sta_eql_vol", fig_save_dir=output_dir, fig_save_type="PNG",
    xtick_spread=3, xtick_label_rotation=90,
    xtick_label_size=12, ytick_label_size=12,
    legend_fontsize=12, legend_loc="upper left"
)
artist.plot()

# plot static min uty / data is from the print output of opt2.py
src_pos_df["子策略一"] = 0.26961431
src_pos_df["子策略二"] = 0.54783229
src_pos_df["子策略三"] = 0.18255340
artist = CPlotBars(
    plot_df=src_pos_df, bar_color=["#00008B", "#4169E1", "#B0C4DE"], stacked=True,
    fig_size=(10, 3), fig_name="weight_sub_strategy_sta_min_uty", fig_save_dir=output_dir, fig_save_type="PNG",
    xtick_spread=3, xtick_label_rotation=90,
    xtick_label_size=12, ytick_label_size=12,
    legend_fontsize=12, legend_loc="upper left"
)
artist.plot()

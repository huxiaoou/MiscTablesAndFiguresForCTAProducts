import os
import pandas as pd
from winterhold2 import CPlotScatter

pd.set_option("float_format", "{:.6f}".format)
pd.set_option("display.width", 0)

input_dir = os.path.join("..", "data", "input")
output_dir = os.path.join("..", "data", "output")

raw_file = "extra.xlsx"
raw_path = os.path.join(input_dir, raw_file)
raw_df = pd.read_excel(raw_path, sheet_name="by_year", dtype={"year": str})
raw_df.set_index("year", inplace=True)
print(raw_df)

artist = CPlotScatter(
    plot_df=raw_df,
    fig_name="scatter_equity_and_commodity_by_year", fig_save_dir=output_dir, fig_save_type="PNG",
    annotations_using_index=True,
    annotations_location_drift=(0.5, 0.5),
    annotations_fontsize=20,
    fig_size=(12, 6.75),
    point_x="南华商品", point_y="中证500",
    point_size=24, point_color="#DC143C",
    xtick_label_size=20, ytick_label_size=20,
    style="seaborn-v0_8-poster",
)
artist.plot()

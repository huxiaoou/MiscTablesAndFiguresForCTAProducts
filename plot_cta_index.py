import os
import pandas as pd
from winterhold2 import CPlotLines

pd.set_option("float_format", "{:.6f}".format)
pd.set_option("display.width", 0)

input_dir = os.path.join("..", "data", "input")
output_dir = os.path.join("..", "data", "output")
performance_indicators = ["hold_period_return",
                          "annual_return", "annual_volatility",
                          "sharpe_ratio", "calmar_ratio",
                          "max_drawdown_scale"]

# load cta index
cta_index_names = ["CTA趋势细分策略指数", "CTA套利细分策略指数", "CTA复合细分策略指数"]
cta_index_df = pd.read_excel(
    os.path.join(input_dir, "朝阳私募指数_1692767524298.xlsx"), sheet_name="中国私募指数",
)[["时间"] + cta_index_names]
cta_index_df.rename(mapper={"时间": "trade_date"}, axis=1, inplace=True)
cta_index_df.set_index("trade_date", inplace=True)

# load nh0100
raw_file = "extra.xlsx"
raw_path = os.path.join(input_dir, raw_file)
raw_df = pd.read_excel(raw_path, sheet_name="daily_ret")
raw_df["Date"] = raw_df["Date"].map(lambda _: _.strftime("%Y-%m-%d"))
raw_df.set_index("Date", inplace=True)

cta_index_df["南华商品"] = (raw_df["NH0100.NHF"] / 100 + 1).cumprod()
cta_index_df = cta_index_df / cta_index_df.iloc[0]
print(cta_index_df)

artist = CPlotLines(
    plot_df=cta_index_df,
    fig_name="cta_index", fig_save_dir=output_dir, fig_save_type="PNG",
    fig_size=(16, 6),
    xtick_count=10, xtick_label_rotation=0,
    style="seaborn-v0_8-poster",
    line_color=['#000080', '#4169E1', '#B0C4DE', '#DC143C', '#4682B4'],
    line_style=['-', '-', '-', '-.', '-'],
    xtick_label_size=16, ytick_label_size=16,
    legend_fontsize=16,
)
artist.plot()

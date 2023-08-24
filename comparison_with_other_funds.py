import os
import pandas as pd
from skyrim.riften import CNAV
from winterhold2 import CPlotLines

pd.set_option("float_format", "{:.6f}".format)
pd.set_option("display.width", 0)

input_dir = os.path.join("..", "data", "input")
output_dir = os.path.join("..", "data", "output")
performance_indicators = ["hold_period_return", "annual_return", "annual_volatility",
                          "sharpe_ratio", "calmar_ratio",
                          "max_drawdown_scale",
                          "q01", "q05"]

adj_return_df = pd.read_csv(
    os.path.join(output_dir, "adj_return.csv"), dtype={"trade_date": str}).set_index("trade_date")
adj_nav_df = (1 + adj_return_df).cumprod()
print(adj_return_df)
print(adj_nav_df)

other_funds_df = pd.read_excel(
    os.path.join(input_dir, "CTA观察池-0630.xlsx"), sheet_name="无公式版",
)[["基金名称", "上海宽德卓越", "九坤量化CTA私募1号", "黑翼CTA-T1", "洛书尊享CTA拾壹号"]]
other_funds_df.rename(mapper={
    "基金名称": "trade_date",
    "上海宽德卓越": "KD",
    "九坤量化CTA私募1号": "JK",
    "黑翼CTA-T1": "HY",
    "洛书尊享CTA拾壹号": "LS",
}, axis=1, inplace=True)
other_funds_df["trade_date"] = other_funds_df["trade_date"].map(lambda z: z.strftime("%Y-%m-%d"))
other_funds_df.set_index("trade_date", inplace=True)
other_funds_df = other_funds_df.replace("查询无数据", None).fillna(method="ffill").astype(float)

merged_df = pd.merge(
    left=adj_nav_df[["GH"]], right=other_funds_df,
    left_index=True, right_index=True,
    how="right"
).fillna(method="ffill").fillna(1)
merged_df = merged_df.loc[merged_df.index < "2023-07-01"]
merged_df = merged_df / merged_df.iloc[0]
print(merged_df)

summary_data = {}
for p in merged_df.columns:
    nav = CNAV(t_raw_nav_srs=merged_df[p], t_annual_rf_rate=0, t_annual_factor=51)
    nav.cal_all_indicators(t_qs=(1, 5))
    summary_data[p] = nav.to_dict(t_type="eng")
summary_df = pd.DataFrame.from_dict(summary_data, orient="index")
summary_df = summary_df[performance_indicators]
print(summary_df)
summary_df.to_csv(os.path.join(output_dir, "comparison_funds_performance.csv"), index_label="funds", float_format="%.2f")

artist = CPlotLines(
    plot_df=merged_df.rename(mapper={"GH": "国海量化"}, axis=1),
    fig_name="comparison_funds_nav", fig_save_dir=output_dir, fig_save_type="PNG",
    fig_size=(19, 4),
    xtick_count=10, xtick_label_rotation=0,
    style="seaborn-v0_8-poster",
    line_color=['#DC143C', '#000080', '#4169E1', '#B0C4DE', '#4682B4'],
    xtick_label_size=16, ytick_label_size=16,
    legend_fontsize=16,
)
artist.plot()

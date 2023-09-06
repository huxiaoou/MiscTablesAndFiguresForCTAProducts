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
                          "max_drawdown_scale"]

merged_df_stp_date = "2023-09-01"

adj_return_df = pd.read_csv(
    os.path.join(output_dir, "optimized-ret-pes-pes-pes.csv.gz"), dtype={"trade_date": str}).set_index("trade_date")
adj_return_df.rename(mapper={"动态效用最优": "GH"}, axis=1, inplace=True)
adj_return_df.index = adj_return_df.index.map(lambda z: "-".join([z[0:4], z[4:6], z[6:8]]))
# adj_nav_df = (1 + adj_return_df).cumprod()
adj_nav_df = pd.DataFrame({
    "GH-30": (adj_return_df["GH"] * 3 + 1).cumprod(),
    "GH-20": (adj_return_df["GH"] * 2 + 1).cumprod(),
    "GH-10": (adj_return_df["GH"] * 1 + 1).cumprod(),
})

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
    os.path.join(output_dir, "performance_GH_by_year.csv"),
    index_label="trade_year", float_format="%.6f",
)
print(summary_by_year_df)

other_funds_df = pd.read_excel(
    os.path.join(input_dir, "CTA观察池-0831.xlsx"), sheet_name="无公式版", header=1,
)[["基金简称", "上海宽德卓越", "九坤量化CTA私募1号", "黑翼CTA-T1", "洛书尊享CTA拾壹号", "永安千象6期基金"]]
other_funds_df.rename(mapper={
    "基金简称": "trade_date",
    "上海宽德卓越": "KD",
    "九坤量化CTA私募1号": "JK",
    "黑翼CTA-T1": "HY",
    "洛书尊享CTA拾壹号": "LS",
    "永安千象6期基金": "QX",
}, axis=1, inplace=True)
other_funds_df.drop(labels=0, inplace=True)
other_funds_df["trade_date"] = other_funds_df["trade_date"].map(lambda z: z.strftime("%Y-%m-%d"))
other_funds_df.set_index("trade_date", inplace=True)
other_funds_df = other_funds_df.replace("查询无数据", None).fillna(method="ffill").astype(float)
other_funds_df = other_funds_df.loc[other_funds_df.index <= "2023-08-01"]

merged_df = pd.merge(
    left=adj_nav_df, right=other_funds_df,
    left_index=True, right_index=True,
    how="right"
).fillna(method="ffill").fillna(1)
merged_df = merged_df.loc[merged_df.index < merged_df_stp_date]
merged_df = merged_df / merged_df.iloc[0]
# print(merged_df)

summary_data = {}
for p in merged_df.columns:
    nav = CNAV(t_raw_nav_srs=merged_df[p], t_annual_rf_rate=0, t_annual_factor=51)
    nav.cal_all_indicators(t_qs=(1, 5))
    summary_data[p] = nav.to_dict(t_type="eng")
summary_df = pd.DataFrame.from_dict(summary_data, orient="index")
summary_df = summary_df[performance_indicators]
summary_df.to_csv(os.path.join(output_dir, "performance_other_funds.csv"), index_label="funds", float_format="%.2f")
print(summary_df)

plot_df: pd.DataFrame = merged_df.rename(mapper={
    "GH-10": "国海量化[10%仓位]",
    "GH-20": "国海量化[20%仓位]",
    "GH-30": "国海量化[30%仓位]",
}, axis=1)
plot_df_since_2022 = plot_df.truncate(before="2022-01-01")
plot_df_since_2022 = plot_df_since_2022 / plot_df_since_2022.iloc[0]

artist = CPlotLines(
    plot_df=plot_df,
    fig_name="nav_with_other_funds", fig_save_dir=output_dir, fig_save_type="PNG",
    fig_size=(19, 4),
    xtick_count=11, xtick_label_rotation=0,
    style="seaborn-v0_8-poster",
    line_color=['#DC143C', '#DB7093', '#FF1493', '#000080', '#4B0082', '#4169E1', '#4682B4', '#B0C4DE'],
    line_style=["-"] * 3 + ["-."] * 5,
    xtick_label_size=16, ytick_label_size=16,
    legend_fontsize=16,
)
artist.plot()

artist = CPlotLines(
    plot_df=plot_df_since_2022,
    fig_name="nav_with_other_funds_since_2022", fig_save_dir=output_dir, fig_save_type="PNG",
    fig_size=(19, 4),
    xtick_count=11, xtick_label_rotation=0,
    style="seaborn-v0_8-poster",
    line_color=['#DC143C', '#DB7093', '#FF1493', '#000080', '#4B0082', '#4169E1', '#4682B4', '#B0C4DE'],
    line_style=["-"] * 3 + ["-."] * 5,
    xtick_label_size=16, ytick_label_size=16,
    legend_fontsize=16,
)
artist.plot()

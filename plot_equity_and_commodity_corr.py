import os
import pandas as pd
from skyrim.riften import CNAV
from winterhold2 import CPlotLines

pd.set_option("float_format", "{:.6f}".format)
pd.set_option("display.width", 0)

input_dir = os.path.join("..", "data", "input")
output_dir = os.path.join("..", "data", "output")

raw_file = "extra.xlsx"
raw_path = os.path.join(input_dir, raw_file)
raw_df = pd.read_excel(raw_path, sheet_name="eq_co_cta")
raw_df["Date"] = raw_df["Date"].map(lambda _: _.strftime("%Y-%m-%d"))
raw_df.set_index("Date", inplace=True)
nav_df: pd.DataFrame = raw_df / raw_df.iloc[0, :]
nav_df.rename(mapper={"NH0100.NHF": "南华商品", "000905.SH": "中证500", "CICSF035.WI": "CTA指数"}, axis=1, inplace=True)
ret_df: pd.DataFrame = nav_df / nav_df.shift(1) - 1
ret_df.fillna(0, inplace=True)
corr_df = raw_df.corr()
nav_df_since_202306 = nav_df.loc[nav_df.index >= "2023-06-01"]
nav_df_since_202306 = nav_df_since_202306 / nav_df_since_202306.iloc[0]

print(raw_df)
print(nav_df)
print(ret_df)
print(corr_df)

cols = ["南华商品", "中证500"]
artist = CPlotLines(
    plot_df=nav_df[cols],
    fig_name="assets_trend", fig_save_dir=output_dir, fig_save_type="PNG",
    fig_size=(16, 4),
    xtick_count=9, xtick_label_rotation=0,
    style="seaborn-v0_8-poster",
    line_color=['#000080', '#4169E1'],
    xtick_label_size=16, ytick_label_size=16,
    legend_fontsize=16,
)
artist.plot()

artist = CPlotLines(
    plot_df=nav_df_since_202306[cols],
    fig_name="assets_trend_since_202306", fig_save_dir=output_dir, fig_save_type="PNG",
    fig_size=(16, 2.8),
    xtick_count=8, xtick_label_rotation=0,
    style="seaborn-v0_8-poster",
    line_color=['#000080', '#4169E1'],
    xtick_label_size=16, ytick_label_size=16,
    legend_fontsize=16,
)
artist.plot()

indicators = ["annual_return", "annual_volatility", "sharpe_ratio", "calmar_ratio", "max_drawdown_scale"]
ret_df["E80C20"] = ret_df[cols] @ pd.Series({"南华商品": 0.2, "中证500": 0.8})
ret_df["E50C50"] = ret_df[cols] @ pd.Series({"南华商品": 0.5, "中证500": 0.5})
ret_df["E20C80"] = ret_df[cols] @ pd.Series({"南华商品": 0.8, "中证500": 0.2})

res_data = {}
for p in ret_df.columns:
    nav = CNAV(t_raw_nav_srs=ret_df[p], t_annual_rf_rate=0, t_annual_factor=252, t_type="RET")
    nav.cal_all_indicators()
    res_data[p] = nav.to_dict(t_type="eng")
res_df = pd.DataFrame.from_dict(res_data, orient="index")
print(res_df[indicators])
res_file = "assets_mix_performance.csv"
res_path = os.path.join(output_dir, res_file)
res_df[indicators].to_csv(res_path, index_label="组合", float_format="%.2f")

simu_nav_df = (ret_df[cols + ["E80C20", "E50C50", "E20C80"]] + 1).cumprod()
artist = CPlotLines(
    plot_df=simu_nav_df,
    fig_name="assets_trend_mix", fig_save_dir=output_dir, fig_save_type="PNG",
    fig_size=(16, 14.5),
    xtick_count=8, xtick_label_rotation=0,
    style="seaborn-v0_8-poster",
    line_color=['#DC143C', '#000080', '#4169E1', '#B0C4DE', '#4682B4'],
    xtick_label_size=20, ytick_label_size=20,
    legend_fontsize=20,
)
artist.plot()

import sys
import os
import pandas as pd
from skyrim.riften import CNAV
from winterhold2 import CPlotLines

pd.set_option("float_format", "{:.6f}".format)
pd.set_option("display.width", 0)

input_dir = os.path.join("..", "data", "input")
output_dir = os.path.join("..", "data", "output")

# --- load raw return
raw_file = "extra.xlsx"
raw_path = os.path.join(input_dir, raw_file)
raw_df = pd.read_excel(raw_path, sheet_name="eq_co_cta")
raw_df["Date"] = raw_df["Date"].map(lambda _: _.strftime("%Y-%m-%d"))
raw_df.set_index("Date", inplace=True)
nav_df: pd.DataFrame = raw_df / raw_df.iloc[0, :]
nav_df.rename(mapper={"NH0100.NHF": "南华商品", "000905.SH": "中证500", "CICSF035.WI": "CTA指数"}, axis=1, inplace=True)
ret_df: pd.DataFrame = nav_df / nav_df.shift(1) - 1
ret_df.fillna(0, inplace=True)
ret_df = ret_df[["中证500"]]

# --- load portfolio return
adj_return_df = pd.read_csv(
    os.path.join(output_dir, "optimized-ret-pes-pes-pes.csv.gz"), dtype={"trade_date": str}).set_index("trade_date")
adj_return_df.rename(mapper={"动态效用最优": "GH"}, axis=1, inplace=True)
adj_return_df.index = adj_return_df.index.map(lambda z: "-".join([z[0:4], z[4:6], z[6:8]]))
adj_nav_df = pd.DataFrame({
    "GH-20": (adj_return_df["GH"] * 2 + 1).cumprod(),
    "GH-10": (adj_return_df["GH"] * 1 + 1).cumprod(),
})

# --- merge return
ret_df["GH-10"] = adj_return_df["GH"] * 1
ret_df["GH-20"] = adj_return_df["GH"] * 2
ret_df = ret_df.truncate(before="2018-01-01")

#
leverage = int(sys.argv[1])
gh_lbl = {1: "GH-10", 2: "GH-20"}[leverage]
cols = [gh_lbl, "中证500"]
indicators = ["hold_period_return", "annual_return", "annual_volatility", "sharpe_ratio", "calmar_ratio", "max_drawdown_scale"]
ret_df["E80C20"] = ret_df[cols] @ pd.Series({gh_lbl: 0.2, "中证500": 0.8})
ret_df["E50C50"] = ret_df[cols] @ pd.Series({gh_lbl: 0.5, "中证500": 0.5})
ret_df["E20C80"] = ret_df[cols] @ pd.Series({gh_lbl: 0.8, "中证500": 0.2})

res_data = {}
for p in ret_df.columns:
    nav = CNAV(t_raw_nav_srs=ret_df[p], t_annual_rf_rate=0, t_annual_factor=252, t_type="RET")
    nav.cal_all_indicators()
    res_data[p] = nav.to_dict(t_type="eng")
res_df = pd.DataFrame.from_dict(res_data, orient="index")[indicators]
res_file = f"equity_mix_portfolio_performance_L{leverage}.csv"
res_path = os.path.join(output_dir, res_file)
res_df.to_csv(res_path, index_label="组合", float_format="%.2f")
print(res_df)

simu_nav_df = (ret_df[["E20C80", "E50C50", "E80C20", "中证500"]] + 1).cumprod()
simu_nav_df.rename(mapper={
    "GH-10": "国海量化[10%仓位]",
    "GH-20": "国海量化[20%仓位]",
    "E80C20": "80%中证500 + 20%国海量化CTA",
    "E50C50": "50%中证500 + 50%国海量化CTA",
    "E20C80": "20%中证500 + 80%国海量化CTA",
}, axis=1, inplace=True)
artist = CPlotLines(
    plot_df=simu_nav_df,
    fig_name=f"equity_mix_portfolio_performance_L{leverage}", fig_save_dir=output_dir, fig_save_type="PNG",
    fig_size=(31, 4.8),
    xtick_count=8, xtick_label_rotation=0,
    style="seaborn-v0_8-poster",
    line_color=['#000080', '#4169E1', '#B0C4DE', '#DC143C'],
    line_style=['-', '-', '-', '-.'],
    xtick_label_size=20, ytick_label_size=20,
    legend_fontsize=20,
)
artist.plot()

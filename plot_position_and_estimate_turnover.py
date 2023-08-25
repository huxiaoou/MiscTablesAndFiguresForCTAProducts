import os

import pandas as pd
from skyrim.falkreath import CLib1Tab1, CTable, CManagerLibReader
from winterhold2 import CPlotBars

output_dir = os.path.join("..", "data", "output")

test_date = "20230330"

sig_lib_id = "A1M020"
sig_lib_dir = "E:\\Deploy\\Data\\Futures\\cta\\signals_opt"
sig_lib_struct = CLib1Tab1(
    t_lib_name=f"{sig_lib_id}.db",
    t_tab=CTable({
        "table_name": "A1M020",
        "primary_keys": {"trade_date": "TEXT", "instrument": "TEXT"},
        "value_columns": {"value": "REAL"},
    })
)

sig_lib_reader = CManagerLibReader(sig_lib_dir, sig_lib_struct.m_lib_name)
sig_lib_reader.set_default(sig_lib_struct.m_tab.m_table_name)
sig_df = sig_lib_reader.read_by_date(t_trade_date=test_date, t_value_columns=["instrument", "value"]).set_index("instrument")
sig_lib_reader.close()
sig_df.rename(mapper={"value": "品种权重"}, axis=1, inplace=True)
print(sig_df)

artist = CPlotBars(
    plot_df=sig_df, bar_color=["#000080"],
    fig_size=(45, 10), fig_name="position", fig_save_dir=output_dir, fig_save_type="PNG",
    xtick_spread=1, xtick_label_rotation=90,
    xtick_label_size=20, ytick_label_size=20,
    legend_fontsize=20, legend_loc="lower center"
)
artist.plot()

sig_lib_reader = CManagerLibReader(sig_lib_dir, sig_lib_struct.m_lib_name)
sig_lib_reader.set_default(sig_lib_struct.m_tab.m_table_name)
sig_df = sig_lib_reader.read_by_conditions(t_conditions=[
    ("trade_date", ">=", "20180101"),
    ("trade_date", "<", "20230801"),
], t_value_columns=["trade_date", "instrument", "value"]).set_index("instrument")
weight_by_date_df = pd.pivot_table(data=sig_df, index="trade_date", columns="instrument", values="value").fillna(0)
print(weight_by_date_df)
dlt_weight = (weight_by_date_df - weight_by_date_df.shift(1)).fillna(0)
print(dlt_weight)
dlt_weight_abs_sum = dlt_weight.abs().sum(axis=1)
daily_turnover_aver = dlt_weight_abs_sum.mean()
annual_turnover = daily_turnover_aver * 252
print(f"daily  turnover = {daily_turnover_aver:7.4f}")
print(f"annual turnover = {annual_turnover    :7.4f}")

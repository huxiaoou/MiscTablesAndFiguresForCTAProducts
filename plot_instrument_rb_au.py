import os
import pandas as pd
from winterhold2 import CPlotLines


def get_instrument_df(p_src_file: str, p_input_dir: str, p_instrument_id: str,
                      p_bgn_date: str, p_end_date: str,
                      ) -> pd.DataFrame:
    src_path = os.path.join(p_input_dir, p_src_file)
    src_df = pd.read_excel(src_path, header=1)
    src_df.rename(mapper={
        "RB.SHF": "螺纹钢",
        "AU.SHF": "黄金",
    }, axis=1, inplace=True)
    src_df["trade_date"] = src_df["Date"].map(lambda _: _.strftime("%Y-%m-%d"))
    src_df.set_index("trade_date", inplace=True)
    src_df = src_df.truncate(before=p_bgn_date, after=p_end_date)
    return src_df[[instrument_id]]


input_dir = os.path.join("..", "data", "input")
output_dir = os.path.join("..", "data", "output")
src_file = "instruments_nav.xlsx"

instrument_id = "螺纹钢"
bgn_date, end_date = "2016-01-01", "2023-08-31"
rb_df = get_instrument_df(p_src_file=src_file, p_input_dir=input_dir, p_instrument_id=instrument_id,
                          p_bgn_date=bgn_date, p_end_date=end_date)
artist = CPlotLines(
    plot_df=rb_df,
    fig_name="instrument_nav_rb", fig_save_dir=output_dir, fig_save_type="PNG",
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

instrument_id = "黄金"
bgn_date, end_date = "2016-01-01", "2023-08-31"
rb_df = get_instrument_df(p_src_file=src_file, p_input_dir=input_dir, p_instrument_id=instrument_id,
                          p_bgn_date=bgn_date, p_end_date=end_date)
artist = CPlotLines(
    plot_df=rb_df,
    fig_name="instrument_nav_au", fig_save_dir=output_dir, fig_save_type="PNG",
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

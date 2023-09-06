import os
import itertools
import numpy as np
import pandas as pd
from skyrim.whiterun import CCalendar, CCalendarMonthly
from skyrim.riften import CNAV
from winterhold2 import CPlotLines


def get_header_df(b_date: str, s_date: str, p_calendar: CCalendar) -> pd.DataFrame:
    iter_dates = p_calendar.get_iter_list(b_date, s_date, True)
    return pd.DataFrame({"trade_date": iter_dates})


def get_return_srs(src_df: pd.DataFrame, header: pd.DataFrame) -> pd.Series:
    ret_df = pd.merge(left=header, right=src_df[["trade_date", "ret"]], on="trade_date", how="left")
    ret_df = ret_df.fillna(0).set_index("trade_date")
    return ret_df["ret"]


def get_qian_ret_srs(src_dir: str, header: pd.DataFrame, ret_type: str) -> pd.Series:
    if ret_type == "pes":
        src_file = "qian-日收益率20230831.xls"
        src_path = os.path.join(src_dir, src_file)
        src_df = pd.read_excel(src_path, header=0, names=["trade_date", "s-ret", "cum-s-ret"])
        src_df["trade_date"] = src_df["trade_date"].map(lambda _: _.strftime("%Y%m%d"))
        src_df["nav"] = src_df["cum-s-ret"] + 1
    else:
        src_file = "qian-原始策略日收益20230821.xls"
        src_path = os.path.join(src_dir, src_file)
        src_df = pd.read_excel(src_path, header=0, names=["trade_date", "s-ret"])
        src_df["trade_date"] = src_df["trade_date"].map(lambda _: _.strftime("%Y%m%d"))
        src_df["nav"] = src_df["s-ret"].cumsum() + 1

    src_df["ret"] = src_df["nav"] / src_df["nav"].shift(1) - 1
    return get_return_srs(src_df, header)


def get_yuex_ret_srs(src_dir: str, header: pd.DataFrame, ret_type: str) -> pd.Series:
    src_file = "PL-yx-0831.xlsx"
    src_path = os.path.join(src_dir, src_file)
    src_df = pd.read_excel(src_path, header=0, names=["trade_date", "ret"])
    impact = 1e-4 if ret_type == "pes" else 0
    src_df["trade_date"] = src_df["trade_date"].map(lambda _: _.replace("/", ""))
    src_df["ret"] = src_df["ret"] - impact
    return get_return_srs(src_df, header)


def get_huxo_ret_srs(src_dir: str, header: pd.DataFrame, ret_type: str) -> pd.Series:
    if ret_type == "pes":
        src_file = "R1M010.HPN001.D00.nav.daily.csv.gz"
    else:
        src_file = "A1M020.HPN001.D00.nav.daily.csv.gz"
    src_path = os.path.join(src_dir, src_file)
    src_df = pd.read_csv(src_path, dtype={"trade_date": str})
    src_df["ret"] = src_df["nav"] / src_df["nav"].shift(1) - 1
    return get_return_srs(src_df, header)


def opt_vanilla(net_ret_df: pd.DataFrame):
    return net_ret_df.mean(axis=1)


def opt_static_eql_vol(net_ret_df: pd.DataFrame):
    net_return_std = net_ret_df.std()
    weight_srs = 1 / net_return_std
    weight_srs = weight_srs / weight_srs.sum()
    print("eql_vol weight:", weight_srs.values)
    adj_return_df: pd.DataFrame = net_ret_df.multiply(weight_srs, axis=1)
    return adj_return_df.sum(axis=1)


def opt_static_min_uty(net_ret_df: pd.DataFrame, p_lbd: float):
    from skyrim.markarth import minimize_utility
    a = 252
    mu, sgm = net_ret_df.mean() * a, net_ret_df.cov() * a
    w, _ = minimize_utility(t_mu=mu.values, t_sigma=sgm.values, t_lbd=p_lbd)
    w_norm = w / np.abs(w).sum()
    print("min_uty weight:", w_norm)
    adj_return_df: pd.DataFrame = net_ret_df.multiply(w_norm, axis=1)
    return adj_return_df.sum(axis=1)


class COptDynamic(object):
    def __init__(self, net_ret_df: pd.DataFrame,
                 p_trn_win: int, p_min_model_days: int,
                 header: pd.DataFrame,
                 p_bgn_date: str, p_stp_date: str, p_calendar: CCalendarMonthly):
        self.net_ret_df = net_ret_df
        self.trn_win, self.min_model_days = p_trn_win, p_min_model_days
        self.header = header
        self.bgn_date, self.stp_date = p_bgn_date, p_stp_date
        self.calendar = p_calendar
        self.model_df = pd.DataFrame()

    def __load_train_dates(self):
        train_dates = []
        iter_months = self.calendar.map_iter_dates_to_iter_months(self.bgn_date, self.stp_date)
        for __train_end_month in iter_months:
            __train_bgn_date, __train_end_date = self.calendar.get_bgn_and_end_dates_for_trailing_window(__train_end_month, self.trn_win)
            train_dates.append((__train_end_month, __train_bgn_date, __train_end_date))
        return train_dates

    def _optimization(self, mu: pd.Series, sgm: pd.DataFrame) -> (np.ndarray, float):
        pass

    def __cal_model_data(self):
        a = 252
        default_weights = pd.Series(data=[1 / 3, 1 / 3, 1 / 3], index=self.net_ret_df.columns)
        model_data = {}
        for train_end_month, train_bgn_date, train_end_date in self.__load_train_dates():
            filter_dates = (self.net_ret_df.index >= train_bgn_date) & (self.net_ret_df.index <= train_end_date)
            ret_df = self.net_ret_df.loc[filter_dates]
            if len(ret_df) < self.min_model_days:
                # print(train_bgn_date, train_end_date, "Not enough dates")
                ws = default_weights
            else:
                if (r0 := np.linalg.matrix_rank(ret_df)) < 3:
                    print(train_end_month, train_bgn_date, train_end_date, f"{r0}/3")
                    ws = None
                else:
                    mu, sgm = ret_df.mean() * a, ret_df.cov() * a
                    w, _ = self._optimization(mu, sgm)
                    ws = pd.Series(data=w, index=mu.index)
            trade_date = self.calendar.get_next_date(train_end_date, 2)
            if (ws_abs_sum := ws.abs().sum()) > 1:
                model_data[trade_date] = ws / ws_abs_sum
            else:
                model_data[trade_date] = ws
        self.model_df = pd.DataFrame.from_dict(model_data, orient="index")
        return 0

    def main(self):
        self.__cal_model_data()
        weight_df = pd.merge(left=self.header, right=self.model_df, left_on="trade_date", right_index=True, how="left")
        weight_df = weight_df.set_index("trade_date").fillna(method="ffill").fillna(method="bfill")
        adj_return_df: pd.DataFrame = self.net_ret_df * weight_df
        return adj_return_df.sum(axis=1)

    def save_model(self, save_dir: str, save_id: str):
        self.model_df.to_csv(
            os.path.join(save_dir, f"model-{save_id}.csv.gz"),
            index_label="trade_date", float_format="%.4f",
        )
        return 0


class COptDynamicWithLambda(COptDynamic):
    def __init__(self, p_lbd: float, **kwargs):
        super().__init__(**kwargs)
        self.lbd = p_lbd


class COptDynamicMinUty(COptDynamicWithLambda):
    def _optimization(self, mu: pd.Series, sgm: pd.DataFrame):
        from markarth2 import minimize_utility_con
        return minimize_utility_con(t_mu=mu.values, t_sigma=sgm.values, t_lbd=self.lbd, t_bounds=(0.1, 0.5), t_pos_lim=(0.4, 1))


class COptDynamicMinUty3(COptDynamicWithLambda):
    def _optimization(self, mu: pd.Series, sgm: pd.DataFrame):
        from skyrim.markarth import minimize_utility_con3
        return minimize_utility_con3(t_mu=mu.values, t_sigma=sgm.values, t_lbd=self.lbd, t_bound=1)


class COptDynamicMinVol(COptDynamic):
    def _optimization(self, mu: pd.Series, sgm: pd.DataFrame):
        from skyrim.markarth import minimize_variance
        return minimize_variance(t_sigma=sgm.values)


def get_base_assets_brief(net_ret_df: pd.DataFrame, performance_indicators: list[str], save_dir: str):
    summary = []
    for asset_id in net_ret_df.columns:
        nav = CNAV(net_ret_df[asset_id], t_annual_rf_rate=0, t_type="RET")
        nav.cal_all_indicators()
        d = nav.to_dict(t_type="eng")
        d.update({"asset_id": asset_id})
        summary.append(d)
    summary_df = pd.DataFrame(summary)
    summary_df.set_index("asset_id", inplace=True)
    summary_save_df = summary_df[performance_indicators]
    summary_save_df.to_csv(
        os.path.join(save_dir, "summary-sub-strategy.csv"),
        index_label="strategy", float_format="%.2f"
    )
    print(summary_save_df)
    return 0


def plot_base_assets_nav(selected_net_ret_df: pd.DataFrame, save_dir: str):
    nav_df = (selected_net_ret_df + 1).cumprod()
    nav_df.index = nav_df.index.map(lambda z: "-".join([z[0:4], z[4:6], z[6:8]]))
    artist = CPlotLines(
        plot_df=nav_df,
        fig_name=f"sub-strategy-nav", fig_save_dir=save_dir, fig_save_type="PNG",
        fig_size=(19, 5),
        xtick_count=10, xtick_label_rotation=0,
        style="seaborn-v0_8-poster",
        line_color=['#000080', '#4169E1', '#B0C4DE', '#DC143C', '#4682B4'],
        xtick_label_size=16, ytick_label_size=16,
        legend_fontsize=16,
    )
    artist.plot()
    return 0


def plot_base_assets_rolling_corr(selected_net_ret_df: pd.DataFrame, win: int, save_dir: str):
    def __get_corr(df: pd.DataFrame, x: str, y: str, w: int):
        x_aver = df[x].rolling(window=w).mean()
        y_aver = df[y].rolling(window=w).mean()
        xy_aver = (df[x] * df[y]).rolling(window=w).mean()
        xx_aver = (df[x] * df[x]).rolling(window=w).mean()
        yy_aver = (df[y] * df[y]).rolling(window=w).mean()
        cov_xy = xy_aver - x_aver * y_aver
        cov_xx = xx_aver - x_aver * x_aver
        cov_yy = yy_aver - y_aver * y_aver
        return cov_xy / np.sqrt(cov_xx * cov_yy)

    def __get_rolling_corr(df: pd.DataFrame) -> pd.DataFrame:
        res = {}
        for v0 in df.columns:
            for v1 in df.columns:
                if v0 < v1:
                    res[f"{v0}和{v1}"] = __get_corr(df, v0, v1, win)
        return pd.DataFrame(res)

    print(selected_net_ret_df.corr())
    adj_ret_rolling_cor = __get_rolling_corr(selected_net_ret_df)
    adj_ret_rolling_cor.index = adj_ret_rolling_cor.index.map(lambda z: "-".join([z[0:4], z[4:6], z[6:8]]))
    artist = CPlotLines(
        plot_df=adj_ret_rolling_cor,
        fig_name="adj_ret_rolling_corr_opt2", fig_save_dir=save_dir, fig_save_type="PNG",
        fig_size=(16, 7),
        xtick_count=9, xtick_label_rotation=0,
        style="seaborn-v0_8-poster",
        line_color=['#000080', '#4169E1', '#B0C4DE'],
        xtick_label_size=16, ytick_label_size=16,
        legend_fontsize=16, legend_loc="lower center",
    )
    artist.plot()
    return 0


def get_opt_assets_brief(net_ret_df: pd.DataFrame, p_lbd: float, performance_indicators: list[str], save_dir: str, **kwargs):
    def __get_ret_statistics(ret_srs: pd.Series, _method_id: str, _comb_id: str) -> dict:
        nav = CNAV(ret_srs, t_annual_rf_rate=0, t_type="RET")
        nav.cal_all_indicators()
        d = nav.to_dict(t_type="eng")
        d.update({"method": _method_id, "comb_id": _comb_id})
        return d

    summary = []
    for ct0, ct1, ct2 in itertools.product(["opt", "pes"], ["opt", "pes"], ["opt", "pes"]):
        selected_cols = [f"qian-{ct0}", f"yuex-{ct1}", f"huxo-{ct2}"]
        comb_id = f"{ct0}-{ct1}-{ct2}"

        opt_vanilla_srs = opt_vanilla(net_ret_df[selected_cols])
        summary.append(__get_ret_statistics(opt_vanilla_srs, "vanilla", comb_id))

        opt_static_eql_vol_srs = opt_static_eql_vol(net_ret_df[selected_cols])
        summary.append(__get_ret_statistics(opt_static_eql_vol_srs, "static_eql_vol", comb_id))

        opt_static_min_uty_srs = opt_static_min_uty(net_ret_df[selected_cols], p_lbd=p_lbd)
        summary.append(__get_ret_statistics(opt_static_min_uty_srs, "static_min_uty", comb_id))

        optimizer = COptDynamicMinUty(p_lbd=p_lbd, net_ret_df=net_ret_df[selected_cols], **kwargs)
        opt_dynamic_min_uty_srs = optimizer.main()
        optimizer.save_model(save_dir=save_dir, save_id=comb_id)
        summary.append(__get_ret_statistics(opt_dynamic_min_uty_srs, "dynami_min_uty", comb_id))

        # optimizer = COptDynamicMinUty3(p_lbd=p_lbd, net_ret_df=net_ret_df[selected_cols], **kwargs) # this model allows short position in sub-assets
        # opt_dynamic_min_uty3_srs = optimizer.main()
        # summary.append(__get_ret_statistics(opt_dynamic_min_uty3_srs, "dynami_min_uty3", comb_id))

        # optimizer = COptDynamicMinVol(net_ret_df=net_ret_df[selected_cols], **kwargs)
        # opt_dynamic_min_vol_srs = optimizer.main()
        # summary.append(__get_ret_statistics(opt_dynamic_min_vol_srs, "dynami_min_vol", comb_id))

        opt_ret_df = pd.DataFrame({
            "简单等权": opt_vanilla_srs,
            "全局等波动": opt_static_eql_vol_srs,
            "全局效用最优": opt_static_min_uty_srs,
            "动态效用最优": opt_dynamic_min_uty_srs,
        })
        opt_nav_df = (opt_ret_df + 1).cumprod()
        opt_nav_df.index = opt_nav_df.index.map(lambda z: "-".join([z[0:4], z[4:6], z[6:8]]))
        artist = CPlotLines(
            plot_df=opt_nav_df,
            fig_name=f"opt-nav-{comb_id}", fig_save_dir=save_dir, fig_save_type="PNG",
            fig_size=(21, 6),
            xtick_count=10, xtick_label_rotation=0,
            style="seaborn-v0_8-poster",
            line_color=['#000080', '#4169E1', '#B0C4DE', '#DC143C', '#4682B4'],
            line_style=['-.'] * 3 + ['-'],
            xtick_label_size=16, ytick_label_size=16,
            legend_fontsize=16,
        )
        artist.plot()

        opt_ret_df.to_csv(os.path.join(save_dir, f"optimized-ret-{comb_id}.csv.gz"),
                          index_label="trade_date", float_format="%.8f")

    summary_df = pd.DataFrame(summary)
    summary_df.set_index(["method", "comb_id"], inplace=True)
    summary_save_df = summary_df[performance_indicators]
    summary_save_df.to_csv(os.path.join(save_dir, "summary-opt-nav.csv"), float_format="%.2f")
    print(summary_save_df)
    return 0


if __name__ == "__main__":
    import sys

    pd.set_option("display.width", 0)
    selected_indicators = ["hold_period_return", "annual_return", "annual_volatility", "sharpe_ratio", "calmar_ratio", "max_drawdown_scale"]

    input_dir = os.path.join("..", "data", "input")
    output_dir = os.path.join("..", "data", "output")
    calendar_path = "E:\\Deploy\\Data\\Calendar\\cne_calendar.csv"
    # calendar_path = "G:\\Database\\Calendar\\cne_calendar.csv"
    bgn_date, stp_date = "20180101", "20230901"
    lbd = float(sys.argv[1])  # suggested value = 25
    trn_win = int(sys.argv[2])  # suggested value = 12
    min_model_days = int(trn_win * 21 * 0.9)
    calendar = CCalendarMonthly(calendar_path)
    header_df = get_header_df(bgn_date, stp_date, calendar)

    qian_pes_srs = get_qian_ret_srs(input_dir, header_df, "pes")
    qian_opt_srs = get_qian_ret_srs(input_dir, header_df, "opt")
    yuex_pes_srs = get_yuex_ret_srs(input_dir, header_df, "pes")
    yuex_opt_srs = get_yuex_ret_srs(input_dir, header_df, "opt")
    huxo_pes_srs = get_huxo_ret_srs(input_dir, header_df, "pes")
    huxo_opt_srs = get_huxo_ret_srs(input_dir, header_df, "opt")
    all_assets_ret_df = pd.DataFrame({
        "qian-pes": qian_pes_srs,
        "qian-opt": qian_opt_srs,
        "yuex-pes": yuex_pes_srs,
        "yuex-opt": yuex_opt_srs,
        "huxo-pes": huxo_pes_srs,
        "huxo-opt": huxo_opt_srs,
    })
    selected_sub_strategy_df = all_assets_ret_df[[
        "qian-pes", "yuex-pes", "huxo-pes"
    ]].rename(mapper={"qian-pes": "子策略一", "yuex-pes": "子策略二", "huxo-pes": "子策略三"}, axis=1)

    get_base_assets_brief(all_assets_ret_df, save_dir=output_dir, performance_indicators=selected_indicators)
    plot_base_assets_nav(selected_net_ret_df=selected_sub_strategy_df, save_dir=output_dir)
    plot_base_assets_rolling_corr(selected_net_ret_df=selected_sub_strategy_df, win=60, save_dir=output_dir)
    get_opt_assets_brief(net_ret_df=all_assets_ret_df, p_lbd=lbd, save_dir=output_dir, performance_indicators=selected_indicators,
                         header=header_df,
                         p_trn_win=trn_win, p_min_model_days=min_model_days,
                         p_bgn_date=bgn_date, p_stp_date=stp_date, p_calendar=calendar)

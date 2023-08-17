import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import platform

this_platform = platform.system().upper()
if this_platform == "WINDOWS":
    # to make Chinese code compatible
    # plt.rcParams["font.family"] = ["sans-serif"]
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False  # 设置正负号


class CPlotBase(object):
    def __init__(self, fig_size: tuple = (16, 9), fig_name: str = "fig_name",
                 style: str = "Solarize_Light2", color_map: str | None = None,
                 fig_save_dir: str = ".", fig_save_type: str = "pdf"):
        self.fig_size = fig_size
        self.fig_name = fig_name
        self.style = style
        self.color_map = color_map
        self.fig_save_dir = fig_save_dir
        self.fig_save_type = fig_save_type

    def _core(self, ax: plt.Axes):
        pass

    def plot(self):
        plt.style.use(self.style)
        fig0, ax0 = plt.subplots(figsize=self.fig_size)
        self._core(ax0)
        fig0_name = self.fig_name + "." + self.fig_save_type
        fig0_path = os.path.join(self.fig_save_dir, fig0_name)
        fig0.savefig(fig0_path, bbox_inches="tight")
        plt.close(fig0)
        return 0


class CPlotAdjustAxes(CPlotBase):
    def __init__(self, title: str = None, title_size: int = 32,
                 xtick_count: int = None, xtick_spread: float = None, xlabel: str = None, xlabel_size: int = 12, xlim: tuple = (None, None),
                 ytick_count: int = None, ytick_spread: float = None, ylabel: str = None, ylabel_size: int = 12, ylim: tuple = (None, None),
                 xtick_label_size: int = 12, xtick_label_rotation: int = 0, xtick_label_font: str = "Times New Roman",
                 ytick_label_size: int = 12, ytick_label_rotation: int = 0, ytick_label_font: str = "Times New Roman",
                 legend_loc: str = "upper left", legend_fontsize: int = 12,
                 **kwargs):
        self.title, self.title_size = title, title_size
        self.xtick_count, self.xtick_spread = xtick_count, xtick_spread
        self.ytick_count, self.ytick_spread = ytick_count, ytick_spread
        self.xlabel, self.ylabel = xlabel, ylabel
        self.xlabel_size, self.ylabel_size = xlabel_size, ylabel_size
        self.xlim, self.ylim = xlim, ylim
        self.xtick_label_size, self.xtick_label_rotation, self.xtick_label_font = xtick_label_size, xtick_label_rotation, xtick_label_font
        self.ytick_label_size, self.ytick_label_rotation, self.ytick_label_font = ytick_label_size, ytick_label_rotation, ytick_label_font
        self.legend_loc, self.legend_fontsize = legend_loc, legend_fontsize
        super().__init__(**kwargs)

    def _set_axes(self, ax: plt.Axes):
        ax.set_title(self.title, fontsize=self.title_size)
        ax.set_xlabel(self.xlabel, fontsize=self.xlabel_size)
        ax.set_ylabel(self.ylabel, fontsize=self.ylabel_size)
        ax.set_xlim(self.xlim[0], self.xlim[1])
        ax.set_ylim(self.ylim[0], self.ylim[1])
        ax.tick_params(axis="x", labelsize=self.xtick_label_size, rotation=self.xtick_label_rotation)
        ax.tick_params(axis="y", labelsize=self.ytick_label_size, rotation=self.ytick_label_rotation)
        ax.legend(loc=self.legend_loc, fontsize=self.legend_fontsize)
        plt.xticks(fontname=self.xtick_label_font)
        plt.yticks(fontname=self.ytick_label_font)
        return 0


class CPlotLines(CPlotAdjustAxes):
    def __init__(self, plot_df: pd.DataFrame,
                 line_width: float = 2, line_style: list = None, line_color: list = None,
                 **kwargs):
        """

        :param plot_df:
        :param line_width:
        :param line_style: one or more ('-', '--', '-.', ':')
        :param line_color: if this parameter is used, then do not use colormap and do not specify colors in line_style
                           str, array-like, or dict, optional
                           The color for each of the DataFrame’s columns. Possible values are:
                           A single color string referred to by name, RGB or RGBA code, for instance ‘red’ or ‘#a98d19’.

                           A sequence of color strings referred to by name, RGB or RGBA code, which will be used for each column recursively.
                           For instance ['green', 'yellow'] each column’s line will be filled in green or yellow, alternatively.
                           If there is only a single column to be plotted, then only the first color from the color list will be used.

                           A dict of the form {column_name:color}, so that each column will be
                           colored accordingly. For example, if your columns are called a and b, then passing {‘a’: ‘green’, ‘b’: ‘red’}
                           will color lines for column 'a' in green and lines for column 'b' in red.

                           short name for color {
                                'b':blue,
                                'g':green,
                                'r':red,
                                'c':cyan,
                                'm':magenta,
                                'y':yellow,
                                'k':black,
                                'w':white,
                            }
        :param kwargs:
        """
        self.plot_df = plot_df
        self.data_len = len(plot_df)
        self.line_width = line_width
        self.line_style = line_style
        self.line_color = line_color
        super().__init__(**kwargs)

    def _set_axes(self, ax: plt.Axes):
        if self.xtick_count:
            xticks = np.arange(0, self.data_len, max(int((self.data_len - 1) / self.xtick_count), 1))
        elif self.xtick_spread:
            xticks = np.arange(0, self.data_len, self.xtick_spread)
        else:
            xticks = None
        if xticks is not None:
            xticklabels = self.plot_df.index[xticks]
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)

        if self.ylim != (None, None):
            y_range = self.ylim[1] - self.ylim[0]
            if self.ytick_count:
                yticks = np.arange(self.ylim[0], self.ylim[1], y_range / self.ytick_count)
            elif self.ytick_spread:
                yticks = np.arange(self.ylim[0], self.ylim[1], self.ytick_spread)
            else:
                yticks = None

            if yticks is not None:
                ax.set_yticks(yticks)
        super()._set_axes(ax)
        return 0

    def _core(self, ax: plt.Axes):
        if self.line_color:
            self.plot_df.plot.line(ax=ax, lw=self.line_width, style=self.line_style if self.line_style else "-", color=self.line_color)
        else:
            self.plot_df.plot.line(ax=ax, lw=self.line_width, style=self.line_style if self.line_style else "-", colormap=self.color_map)
        self._set_axes(ax)
        return 0


if __name__ == "__main__":
    n = 252 * 5
    df = pd.DataFrame({
        "T": [f"T{_:03d}" for _ in range(n)],
        "上证50": np.cumprod((np.random.random(n) * 2 - 1) / 100 + 1),
        "沪深300": np.cumprod((np.random.random(n) * 2 - 1) / 100 + 1),
        "中证500": np.cumprod((np.random.random(n) * 2 - 1) / 100 + 1),
        "南华商品": np.cumprod((np.random.random(n) * 2 - 1) / 100 + 1),
        "TEST": np.random.random(n) * 2 - 1,
    }).set_index("T")
    print(df.tail())

    arist = CPlotLines(
        plot_df=df, fig_name="test", style="seaborn-v0_8-poster",
        # xtick_count=10, xtick_label_rotation=90,
        # ylim=(0.5, 2.1), ytick_spread=0.25,
        ylim=(-1, 2.1), ytick_count=10,
        # line_style=['-', '--', '-.', ':', ],
        # line_width=2,
        # line_color=['r', 'g', 'b'],
        line_color=['#A62525', '#188A06', '#06708A', '#DAF90E'],
        # color_map="winter",
        xtick_label_size=16, ytick_label_size=16,
        # title="指数走势", xlabel='xxx', ylabel='yyy',
    )
    arist.plot()

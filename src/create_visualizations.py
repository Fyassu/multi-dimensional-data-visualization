"""
Script to generate multi-dimensional data visualizations for the
CMP_SC-8630 data visualization assignment.
"""

import os
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import numpy as np


def ensure_output_dir(path: str) -> None:
    """Ensure that the output directory exists."""
    os.makedirs(path, exist_ok=True)

def plot_weather_heatmap(df: pd.DataFrame, outdir: str) -> str:
    """Vẽ heatmap nhiệt độ trung bình theo thành phố và tháng."""

    # Tính nhiệt độ trung bình theo từng thành phố/tháng
    temp_matrix = (
        df.groupby(["city", "month"])["avg_temp"]
        .mean()
        .unstack()
        .sort_index(axis=1)
    )

    # Khởi tạo figure
    fig, ax = plt.subplots(figsize=(10, 4))

    # Vẽ heatmap
    sns.heatmap(
        temp_matrix,
        cmap="coolwarm",
        annot=True,
        fmt=".1f",
        cbar_kws={"label": "Nhiệt độ trung bình"},
        ax=ax,
    )

    ax.set_title("Nhiệt độ trung bình hàng tháng theo thành phố")
    ax.set_xlabel("Tháng")
    ax.set_ylabel("Thành phố")

    out_path = os.path.join(outdir, "weather_heatmap.png")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    return out_path


def plot_weather_scatter(df: pd.DataFrame, outdir: str) -> str:
    """Biểu đồ scatter thể hiện nhiệt độ, độ ẩm và lượng mưa."""

    data = df.copy()

    # Chuyển lượng mưa sang dạng số, với các giá trị null hoặc không hợp lệ được coi là 0
    data["precip"] = (
        pd.to_numeric(data["precip"], errors="coerce")
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    min_size, max_size = 20, 300

    # Scatter plot chính
    sns.scatterplot(
        data=data,
        x="avg_humidity",
        y="avg_temp",
        hue="city",
        size="precip",
        sizes=(min_size, max_size),
        alpha=0.65,
        legend=False,
        ax=ax,
    )

    ax.set_xlabel("Độ ẩm trung bình")
    ax.set_ylabel("Nhiệt độ trung bình")
    ax.set_title(
        "Thời tiết"
    )

    # Legend cho màu sắc (thành phố)
    palette = sns.color_palette()

    city_handles = [
        mlines.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color="w",
            markerfacecolor=palette[i],
            markersize=8,
            label=city,
        )
        for i, city in enumerate(data["city"].unique())
    ]

    city_legend = ax.legend(
        handles=city_handles,
        title="City",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    ax.add_artist(city_legend)

    # Legend cho kích thước marker (lượng mưa)
    max_precip = data["precip"].max()

    size_values = np.linspace(0, max_precip, 4)

    size_handles = [
        plt.scatter(
            [],
            [],
            s=np.interp(
                value,
                [0, max_precip],
                [min_size, max_size]
            ),
            color="gray",
            alpha=0.65,
            label=f"{value:.2f}",
        )
        for value in size_values
    ]

    ax.legend(
        handles=size_handles,
        title="Lượng mưa",
        bbox_to_anchor=(1.02, 0),
        loc="lower left",
    )

    out_path = os.path.join(outdir, "weather_scatter.png")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    return out_path

def plot_global_temp_heatmap(df: pd.DataFrame, outdir: str) -> str:
    """Heatmap thể hiện sai lệch nhiệt độ toàn cầu theo năm và tháng."""

    months = [
        "Jan", "Feb", "Mar", "Apr",
        "May", "Jun", "Jul", "Aug",
        "Sep", "Oct", "Nov", "Dec"
    ]

    # Chuyển dữ liệu từ wide format sang long format
    temp_long = df.melt(
        id_vars="Year",
        value_vars=months,
        var_name="Month",
        value_name="Anomaly",
    )

    # Đánh số tháng để dễ sắp xếp
    month_map = {
        month: idx + 1
        for idx, month in enumerate(months)
    }

    temp_long["MonthNum"] = temp_long["Month"].map(month_map)

    # Tạo ma trận Year x Month
    heatmap_data = (
        temp_long
        .pivot(
            index="Year",
            columns="MonthNum",
            values="Anomaly",
        )
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        heatmap_data,
        cmap="coolwarm",
        vmin=-1.5,
        vmax=1.5,
        linewidths=0,
        cbar_kws={
            "label": (
                "Sai lệch nhiệt độ (so với 1951–1980)"
            )
        },
        ax=ax,
    )

    ax.set_xticks(np.arange(len(months)) + 0.5)
    ax.set_xticklabels(months, rotation=45)

    ax.set_title(
        "Nhiệt độ toàn cầu theo tháng và năm (1880–2023)"
    )

    ax.set_xlabel("Month")
    ax.set_ylabel("Year")

    out_path = os.path.join(
        outdir,
        "global_temp_heatmap.png"
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    return out_path

def plot_minnesota_precip_line(
    df: pd.DataFrame,
    outdir: str
) -> str:
    """Biểu đồ đường lượng mưa theo thời gian cho từng trạm."""

    data = df.copy()

    # Ghép năm + tháng thành cột datetime
    data["date"] = pd.to_datetime(
        {
            "year": data["year"],
            "month": data["mo"],
            "day": 1,
        }
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    # Vẽ đường cho từng trạm đo
    sns.lineplot(
        data=data,
        x="date",
        y="precip",
        hue="site",
        ax=ax,
    )

    ax.set_title(
        "Lượng mưa hàng tháng theo trạm ở Minnesota (1927–1936)"
    )

    ax.set_xlabel("Năm")
    ax.set_ylabel("Lượng mưa")

    ax.legend(
        title="Trạm",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    out_path = os.path.join(
        outdir,
        "minnesota_precip_line.png"
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    return out_path


def main() -> List[str]:
    """Run all visualizations and return a list of generated file paths."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    out_dir = os.path.join(base_dir, "output")
    ensure_output_dir(out_dir)
    figures: List[str] = []

    # Load and plot weather data
    weather_path = os.path.join(data_dir, "weather_data.csv")
    weather_df = pd.read_csv(weather_path)
    figures.append(plot_weather_heatmap(weather_df, out_dir))
    figures.append(plot_weather_scatter(weather_df, out_dir))

    # Load and plot global temperature anomalies
    global_path = os.path.join(data_dir, "global_temp.csv")
    global_df = pd.read_csv(global_path, skiprows=1)
    global_df = global_df.replace("***", pd.NA)
    for col in global_df.columns[1:]:
        global_df[col] = pd.to_numeric(global_df[col], errors="coerce")
    figures.append(plot_global_temp_heatmap(global_df, out_dir))

    # Load and plot Minnesota weather data
    minn_path = os.path.join(data_dir, "minnesota_weather.csv")
    minn_df = pd.read_csv(minn_path)
    figures.append(plot_minnesota_precip_line(minn_df, out_dir))

    return figures


if __name__ == "__main__":
    generated = main()
    print("Generated figures:")
    for path in generated:
        print(path)
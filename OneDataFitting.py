# 20240613094359 做单次拟合的程序，用于处理弹簧松弛实验数据
# 基于【5】中用的函数OneDataFitting
# 要求无表头，仅两列数据，第一列为x，第二列为y
# 并将拟合参数，决定系数等几个拟合关键结果输出到交互界面中
# 为提高效率，可以在程序中选择筛选数据，包括数据上下限截断，数据基于label的截断和数据基于label的剔除，并调整数据轴上下限。

import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np


# 定义目标函数为指数衰减函数
# 这里拟合后的目标函数只需要在这里修改，其他的都未调用，如果增减参数量也只用在此修改，但还需要修改后面的参数约束范围
def objective_function(x, a, b, c, d, e):
    return a + b * np.exp(- x / c)

def read_data(file_path, sheet_name):
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    x_data = data.iloc[:, 0].values
    y_data = data.iloc[:, 1].values
    return x_data, y_data

# 该函数用于截断数据
def truncate_data(x_data, y_data, x_min, x_max, y_min, y_max, index_min, index_max):
    original_indices = np.arange(len(x_data))
    mask = (x_data >= x_min) & (x_data <= x_max) & (y_data >= y_min) & (y_data <= y_max)
    x_data = x_data[mask]
    y_data = y_data[mask]
    original_indices = original_indices[mask]
    index_range = np.arange(index_min, min(index_max, len(x_data)))
    return x_data[index_range], y_data[index_range], original_indices[index_range]

# 该函数用于移除指定点
def remove_points(x_data, y_data, original_indices, remove_indices):
    """
    根据原始数据序号列表剔除指定的多个点。
    """
    # 转换为布尔数组，初始时所有值都为True
    keep_mask = np.ones(len(x_data), dtype=bool)

    # 对于每个要剔除的序号，将对应的布尔值设置为False
    for remove_index in remove_indices:
        # 查找并标记要剔除的点
        remove_index_in_truncated = np.where(original_indices == remove_index)[0]
        if remove_index_in_truncated.size > 0:
            keep_mask[remove_index_in_truncated] = False

    # 使用布尔索引数组保留未被剔除的点
    x_data_kept = x_data[keep_mask]
    y_data_kept = y_data[keep_mask]
    original_indices_kept = original_indices[keep_mask]

    return x_data_kept, y_data_kept, original_indices_kept


def fit_data_and_evaluate(x_data, y_data, original_indices):
    initial_guess = [1, 1, 1, 1, 1]
    popt, _ = curve_fit(objective_function, x_data, y_data, bounds=([0, 0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf]),
                        p0=initial_guess, maxfev=10000)
    y_pred = objective_function(x_data, *popt)
    relative_errors = np.abs((y_data - y_pred) / y_data)
    max_relative_error_index = np.argmax(relative_errors)
    max_relative_error = relative_errors[max_relative_error_index]
    max_relative_error_original_index = original_indices[max_relative_error_index]
    max_relative_error_x_value = x_data[max_relative_error_index]

    # 计算统计指标
    ss_res = np.sum((y_data - y_pred) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    mse = np.mean((y_data - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_data - y_pred))

    return popt, r_squared, mse, rmse, mae, max_relative_error, max_relative_error_original_index, max_relative_error_x_value


def plot_data(x_data, y_data, popt, y_limits=None, error_y_limits=None, title=None):
    fig, ax1 = plt.subplots()

    # 绘制原始数据和拟合曲线
    ax1.scatter(x_data, y_data, label='Original Data', color='blue')
    x_line = np.arange(min(x_data), max(x_data), 0.001)
    y_line = objective_function(x_line, *popt)
    ax1.plot(x_line, y_line, '--', color='red', label='Fitted Curve', linewidth=3)
    ax1.set_xlabel('Strain/1')
    ax1.set_ylabel('Stress/MPa', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # 设置主y轴（左侧）范围
    if y_limits is not None:
        ax1.set_ylim(y_limits)

    # 计算并绘制相对误差
    ax2 = ax1.twinx()
    y_pred = objective_function(x_data, *popt)
    relative_errors = np.abs((y_data - y_pred) / y_data) * 100  # 相对误差，百分比形式
    ax2.plot(x_data, relative_errors, ':', color='green', label='Relative Error', linewidth=3)
    ax2.set_ylabel('Relative Error (%)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    # 设置相对误差y轴（右侧）范围
    if error_y_limits is not None:
        ax2.set_ylim(error_y_limits)

    # 创建图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    if title:
        plt.title(title)

    fig.tight_layout()
    plt.show()

    return relative_errors


def main():
    file_path = '2024.06.13-测试数据.xlsx'  # 替换为实际文件路径
    sheet_name = '测试数据'
    x_data, y_data = read_data(file_path, sheet_name)
    x_min, x_max, y_min, y_max = 0, 12000, 0, 20
    index_min, index_max = 0, 10000
    x_data_truncated, y_data_truncated, original_indices_truncated = truncate_data(x_data, y_data, x_min, x_max, y_min,
                                                                                   y_max, index_min, index_max)

    # 假设我们要剔除的原始序号列表
    remove_indices = [0]  # 根据实际需求调整
    x_data_final, y_data_final, original_indices_final = remove_points(x_data_truncated, y_data_truncated,
                                                                       original_indices_truncated, remove_indices)

    popt, r_squared, mse, rmse, mae, max_relative_error, max_relative_error_original_index, max_relative_error_x_value = fit_data_and_evaluate(
        x_data_final, y_data_final, original_indices_final)

    print(f"拟合得到的参数：{popt}")
    print(f"决定系数R^2：{r_squared}")
    print(f"均方误差MSE：{mse}")
    print(f"均方根误差RMSE：{rmse}")
    print(f"平均绝对误差MAE：{mae}")
    print(f"最大相对误差：{max_relative_error * 100}%")
    print(f"最大相对误差对应的截断前序号（从0开始）：{max_relative_error_original_index}")
    print(f"最大相对误差对应的x值：{max_relative_error_x_value}")

    title = "Fitting-Result"

    plot_data(x_data_final, y_data_final, popt, title=title)

    # 计算拟合曲线的Y值和相对误差
    y_pred_final = objective_function(x_data_final, *popt)
    relative_errors_final = np.abs((y_data_final - y_pred_final) / y_data_final) * 100

    # 准备DataFrame中的数据，确保所有列长度相等
    max_length = max(len(x_data), len(x_data_final))
    data_dict = {
        'Original X': pd.Series(x_data).reindex(range(max_length)),
        'Original Y': pd.Series(y_data).reindex(range(max_length)),
        'Final X': pd.Series(x_data_final).reindex(range(max_length)),
        'Final Y': pd.Series(y_data_final).reindex(range(max_length)),
        'Predicted Y': pd.Series(y_pred_final).reindex(range(max_length)),
        'Relative Error (%)': pd.Series(relative_errors_final).reindex(range(max_length))
    }
    #
    # # 创建一个DataFrame来存储所有数据和结果
    # results_df = pd.DataFrame(data_dict)
    #
    # # 准备拟合参数和计算结果参数的DataFrame
    # parameters_dict = {
    #     'Parameter': ['a', 'b', 'c', 'R^2', 'MSE', 'RMSE', 'MAE', 'Max Relative Error (%)', 'Max Relative Error X', 'Max Relative Error Original Index'],
    #     'Value': [*popt, r_squared, mse, rmse, mae, max_relative_error, max_relative_error_x_value, max_relative_error_original_index]
    # }
    # parameters_df = pd.DataFrame(parameters_dict)
    # # 将所有结果写入同一个Excel文件的不同部分
    # with pd.ExcelWriter('AnalysisResults.xlsx') as writer:
    #     results_df.to_excel(writer, sheet_name='Data and Errors', index=False)
    #     parameters_df.to_excel(writer, sheet_name='Parameters and Stats', index=False)


if __name__ == "__main__":
    main()


#  20240613105556 本程序用于实现批量拟合弹簧松弛数据
# 将弹簧上升段和下降段分离，本函数主要用于处理弹簧数据，做了针对性适配
# 删除弹簧部分，可以用于大部分的批量拟合，只需要针对性处理数据就行

# 引用区域---------------------------------------------------------------------------------------------------------------
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import inspect
import os
from datetime import datetime


# 定义目标函数------------------------------------------------------------------------------------------------------------
# 将目标函数拆成多部分，为了后期区分不同部分的影响
def objective_function(x, a, b, c, d, f):
    return function_part1(a) + function_part2(x, b, c) + function_part3(x, d, f)

def function_part1(a):
    return a

def function_part2(x, b, c):
    return b * np.exp(- x / c)

def function_part3(x, d, f):
    return d * np.exp(- x / f)


# 数据提取和处理----------------------------------------------------------------------------------------------------------
def extract_and_split_data(file_path, sheet_name, n, name_line, x_col, y_col):
    # 通过excel地址和sheet名，提取该sheet中所有数据
    # 通过参数n，将提取并拆分成以n列为一组的数据集
    # 通过参数name_line，认为前name_line行为表头，将每一组数据集中不包括表头的实际数据提取，表头中的第一行第一列为数据标签
    # 通过x_col和y_col，确定需要的x和y数据对应每一组数据的列
    # 输出每一组的标签，x数据和y数据

    # 读取指定的sheet中的数据
    data = pd.read_excel(file_path, sheet_name=sheet_name, header=None)     # 这个headers的指定非常重要，不指定会认为第一行为索引

    # 提取表头
    headers = data.iloc[:name_line, :]

    # 初始化结果
    groups = []

    # 分组处理数据，输出每组的label，x和y
    for i in range(0, data.shape[1], n):
        group = data.iloc[name_line:, i:i + n].dropna()
        if group.shape[1] >= max(x_col, y_col):
            label = headers.iloc[0, i] if headers.shape[1] > i else 'Unknown'
            x_data = group.iloc[:, x_col - 1].values.astype(float)
            y_data = group.iloc[:, y_col - 1].values.astype(float)
            groups.append((label, x_data, y_data))
        else:
            print("Input X/Y column is large than the total column")

    return groups

# 用于处理弹簧数据，将随时间的回弹力增长和下降段分开
def split_growth_decline(data_groups):
    # 对给定数据，形式为（label，x_data, y_data)
    # 已知其增长形式为y随x先增后减，处理数据，将y随x增长部分和y随x减小部分分开
    # 输出标签，增长部分xy，减小部分xy
    result = []

    for label, x_data, y_data in data_groups:
        # 找到y的最大值索引
        max_index = y_data.argmax()

        # 分割数据成增长部分和减小部分
        x_growth = x_data[:max_index + 1]
        y_growth = y_data[:max_index + 1]
        x_decline = x_data[max_index:]
        y_decline = y_data[max_index:]

        result.append((label, x_growth, y_growth, x_decline, y_decline))
        # result.append((label, x_decline, y_decline))

    return result


# 数据拟合---------------------------------------------------------------------------------------------------------------
def auto_fit(x_data, y_data):
    # 获取目标函数的参数信息
    params = inspect.signature(objective_function).parameters
    num_params = len(params) - 1  # 除去x参数
    initial_guess = [1] * num_params
    lower_bounds = [0] * num_params
    upper_bounds = [np.inf] * num_params

    # 使用curve_fit进行数据拟合
    popt, pcov = curve_fit(objective_function, x_data, y_data, p0=initial_guess,
                           bounds=(lower_bounds, upper_bounds), maxfev=1000000)

    # 预测y值
    y_pred = objective_function(x_data, *popt)

    # 计算残差平方和
    ss_res = np.sum((y_data - y_pred) ** 2)

    # 计算总平方和
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)

    # 计算R平方
    r_squared = 1 - (ss_res / ss_tot)

    # 计算均方误差
    mse = np.mean((y_data - y_pred) ** 2)

    return popt, mse, r_squared, y_pred, x_data, y_data

# 批量拟合
def batch_fit(split_data):
    # 以split_growth_decline()中的result.append((label, x_growth, y_growth, x_decline, y_decline))为输入参数
    # 把每组的x_decline 和 y_decline输入到auto_fit函数中，
    # 输出每组拟合结果，label，popt, mse, r_squared, y_pred, x_data, y_data

    fit_results = []

    for label, x_growth, y_growth, x_decline, y_decline in split_data:
        popt, mse, r_squared, y_pred, x_data, y_data = auto_fit(x_decline, y_decline)
        fit_results.append((label, popt, mse, r_squared, y_pred, x_data, y_data))

    return fit_results


# 数据绘图---------------------------------------------------------------------------------------------------------------
# 输入行数和列数，用于确定绘制组合图的行列数
# 绘制一个组合图，多行多列的组合图
# 在每个图中绘制一组拟合结果
def plot_group_fits(subplot_row, subplot_col, fit_results, flag=0):
    output_dir = 'OutputData'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, axes = plt.subplots(subplot_row, subplot_col, figsize=(15, 10))
    fig.suptitle(datetime.now().strftime("%Y.%m.%d-%H%M%S"), fontsize=10, verticalalignment='top',
                 horizontalalignment='center')

    axes = axes.flatten()       # 这句代码的意思是将axes这个多维数组（比如m*n的二维数组）转换为总数不变的一维数组，方便遍历

    for i, (label, popt, mse, r_squared, y_pred, x_data, y_data) in enumerate(fit_results):
        if i >= len(axes):
            break
        ax = axes[i]

        ax.scatter(x_data, y_data, label='Original Data', s=10, linewidths=0.5, color='#1f77b4')
        ax.plot(x_data, y_pred, '--', color='#ff0000', linewidth=2, label='Fit')

        ax.set_title(label)
        # ax.set_xlabel('X Data')
        # ax.set_ylabel('Y Data')

        if flag == 0:
            relative_errors = np.abs((y_data - y_pred) / y_data) * 100
            ax2 = ax.twinx()
            ax2.set_zorder(ax.get_zorder() - 1)
            ax.patch.set_visible(False)
            ax2.plot(x_data, relative_errors, ':', color='green', label='Relative Error', linewidth=1)
            ax2.set_ylabel('Relative Error (%)')
            ax2.legend(loc='upper right')

            # 合并图例，为了让两个轴的图例用一个显示
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc='upper right')

        else:
            ax.legend(loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_file = os.path.join(output_dir, datetime.now().strftime("%Y.%m.%d-%H%M%S") + 'FittingFigure.png')
    plt.savefig(output_file, dpi=600)   # , transparent=True # 这句话在论文里有用，导出图片不能看
    plt.show()


# 结果数据批量导出--------------------------------------------------------------------------------------------------------
def save_fit_results_to_excel(output_file_name, fit_results):
    # 定义输出文件路径
    output_dir = 'OutputData'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file_path = os.path.join(output_dir, datetime.now().strftime("%Y.%m.%d-%H%M%S-") + output_file_name)

    # 获取目标函数的参数信息
    params = inspect.signature(objective_function).parameters
    param_names = list(params.keys())[1:]  # 排除第一个参数x

    parameters_summary = []

    # 创建一个字典来存储各个DataFrame，以便按顺序写入Excel
    sheets = {}

    for label, popt, mse, r_squared, y_pred, x_data, y_data in fit_results:
        data = {
            'X': x_data,
            'Y Original': y_data,
            'Y Fitted': y_pred,
            'MSE': mse,
            'R^2': r_squared
        }

        # 添加动态参数列
        for i, param_name in enumerate(param_names):
            data[param_name] = popt[i]

        df = pd.DataFrame(data)
        sheets[label[:31]] = df

        # 汇总参数
        param_summary = {'Label': label, 'MSE': mse, 'R^2': r_squared}
        for i, param_name in enumerate(param_names):
            param_summary[param_name] = popt[i]
        parameters_summary.append(param_summary)

    # 参数汇总表放在第一位
    sheets['Parameters Summary'] = pd.DataFrame(parameters_summary)

    # 将所有DataFrame写入Excel文件
    with pd.ExcelWriter(output_file_path) as writer:
        # 先写入参数汇总表
        sheets['Parameters Summary'].to_excel(writer, sheet_name='Parameters Summary', index=False)
        # 再写入其他表
        for sheet_name, df in sheets.items():
            if sheet_name != 'Parameters Summary':
                df.to_excel(writer, sheet_name=sheet_name, index=False)


# 主函数----------------------------------------------------------------------------------------------------------------
def main():
    file_path = "2024.06.06-数据整理.xlsx"  # 数据来源表格（要求在同一根目录下）
    sheet_name = "测试数据"              # 数据来源sheet
    n = 4                               # 每组数据列数
    name_line = 3                       # 表头行数
    x_col = 1                           # 需要的x数据在每组数据中的排序（从1开始）
    y_col = 4                           # 需要的y数据在每组数据中的排序（从1开始）
    subplot_row, subplot_col = 5, 3     # 希望绘制的组图行列数
    output_file_name = "弹簧第一次数据拟合汇总.xlsx"   # 输出文件名称，会在OutputData中输出文件和图片，用文件名的时间区分

    excel_data = extract_and_split_data(file_path, sheet_name, n, name_line, x_col, y_col)
    split_data = split_growth_decline(excel_data)
    fit_results = batch_fit(split_data)
    plot_group_fits(subplot_row, subplot_col, fit_results)
    print("Fitting finished")
    save_fit_results_to_excel(output_file_name, fit_results)
    print("Save finished")


if __name__ == "__main__":
    main()

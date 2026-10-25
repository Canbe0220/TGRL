import os
import re
import random
import numpy as np


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def batch_modify_fjsp_instances(input_dir, output_dir, num_fac):
    """批量读取 FJSP 实例，添加工厂数量、运输周期、机器工厂映射和距离矩阵，并规范化命名。"""
    
    trans_intra_min = 0
    trans_intra_max = 50
    trans_inter_min = 50
    trans_inter_max = 99
    shuttle_period_min = 50
    shuttle_period_max = 99

    # 如果输出文件夹不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历文件夹内的所有文件
    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)

        # 跳过子文件夹，只处理文件
        if not os.path.isfile(filepath):
            continue

        with open(filepath, "r") as f:
            # 过滤掉空行
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            continue

        # 解析第一行并生成新Header
        header_parts = lines[0].split()
        num_jobs = int(header_parts[0])
        num_mas = int(header_parts[1])

        # 原实例的第3个参数通常是 avg_options_per_ope，如果没有则默认为 1.0
        avg_options = header_parts[2] if len(header_parts) > 2 else "1.0"

        # 随机生成班车周期
        shuttle_period = random.randint(shuttle_period_min, shuttle_period_max)

        # 新的第一行格式: num_jobs num_mas num_fac shuttle_period avg_options
        new_header = (
            f"{num_jobs} {num_mas} {num_fac} {shuttle_period} {avg_options}"
        )

        # 生成机器-工厂映射 (Map)
        mas_fac_assignment = sorted([m % num_fac for m in range(num_mas)])
        fac_mapping_line = " ".join(map(str, mas_fac_assignment))

        # 生成距离矩阵
        new_total_nodes = num_mas + 2
        distance_matrix = [
            [0] * new_total_nodes for _ in range(new_total_nodes)
        ]

        # 机器间的距离
        for i in range(1, num_mas + 1):
            for j in range(i + 1, num_mas + 1):
                if mas_fac_assignment[i - 1] == mas_fac_assignment[j - 1]:
                    dist = random.randint(trans_intra_min, trans_intra_max)
                else:
                    dist = random.randint(trans_inter_min, trans_inter_max)
                distance_matrix[i][j] = dist
                distance_matrix[j][i] = dist

        # 机器到起终点的距离
        for i in range(1, num_mas + 1):
            dist_start = random.randint(trans_intra_min, trans_intra_max)
            distance_matrix[0][i] = dist_start
            distance_matrix[i][0] = dist_start

            dist_end = random.randint(trans_intra_min, trans_intra_max)
            distance_matrix[i][new_total_nodes - 1] = dist_end
            distance_matrix[new_total_nodes - 1][i] = dist_end

        # 起点到终点的距离
        distance_matrix[0][new_total_nodes - 1] = 0
        distance_matrix[new_total_nodes - 1][0] = 0

        # 精确重构文件名
        base, ext = os.path.splitext(filename)
        match = re.search(r'_(\d+)$', base)

        if match:
            # 拿到末尾的数字部分 (例如 '001')
            idx_str = match.group(1)
            # 截取掉末尾的 '_001'，保留前面的主体部分
            base_without_idx = base[:match.start()]
            # 组合成: 主体_工厂_序号.后缀
            new_filename = f"{base_without_idx}_{num_fac}f_{idx_str}{ext}"
        else:
            # 末尾没有下划线加数字 (例如 '3128_10j_5m' 或 '10j_5m')，直接在末尾追加
            new_filename = f"{base}_{num_fac}f{ext}"

        # 写入新文件
        out_filepath = os.path.join(output_dir, new_filename)
        with open(out_filepath, "w") as f:
            # 写入修改后的第一行
            f.write(new_header + "\n")

            # 写入原有的工艺路线数据 (第 1 行到第 num_jobs 行)
            for line in lines[1 : num_jobs + 1]:
                f.write(line + "\n")

            # 写入 Map
            f.write(fac_mapping_line + "\n")

            # 写入距离矩阵
            for row in distance_matrix:
                f.write(" ".join(map(str, row)) + "\n")

    print(f"处理完成！所有文件已成功保存至: {output_dir}")


# ===== 运行入口 =====
if __name__ == "__main__":

    setup_seed(200)
    input_path = "Public"
    output_path = "data_test/Public_03"

    batch_modify_fjsp_instances(input_path, output_path, num_fac=3)
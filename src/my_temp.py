import pandas as pd

source_file = "C:\\Users\86155\Desktop\PythonProject\data\processed\A_final_input.csv"
target_file = "C:\\Users\86155\Desktop\PythonProject\\user_data\ggpower\\raw\\1.csv"
target_pid = "SHTF365CHSGHWDFF"

# 根据实际文件类型选择读取方式
# 如果是 CSV: df = pd.read_csv(source_file)
# 如果是 TSV: df = pd.read_csv(source_file, sep='\t')
# 如果是 JSON Lines: df = pd.read_json(source_file, lines=True)
try:
    # 以 CSV 为例，请根据实际情况修改
    df = pd.read_csv(source_file)  # 若为 TSV 加 sep='\t'

    # 筛选 pid 列
    filtered = df[df['pid'] == target_pid]

    # 仅保留所需字段
    result = filtered[['description', 'title', 'pid']]

    # 保存为 CSV（或 JSON 等）
    result.to_csv(target_file, index=False)
    print(f"筛选完成，共 {len(result)} 条记录，保存至 {target_file}")
except FileNotFoundError:
    print(f"文件 {source_file} 未找到")
except KeyError as e:
    print(f"文件中缺少列: {e}")
except Exception as e:
    print(f"发生错误: {e}")

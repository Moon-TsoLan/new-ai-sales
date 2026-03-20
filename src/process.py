# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 17:08:49 2026

@author: 86155
"""

import pandas as pd
import ast
import numpy as np
from pathlib import Path

def extract_material_color(product_details_str):
    """
    从product_details字符串中提取材质(Fabric)和颜色(Color)
    product_details_str格式: "[{'Style Code': '...'}, {'Fabric': '...'}, ...]"
    返回: (材质, 颜色) 元组，若缺失则返回 (None, None)
    """
    material = None
    color = None
    if pd.isna(product_details_str) or product_details_str == '':
        return material, color
    try:
        # 将字符串解析为Python列表
        details_list = ast.literal_eval(product_details_str)
        if not isinstance(details_list, list):
            return material, color
        for item in details_list:
            if isinstance(item, dict):
                # 查找Fabric键（不区分大小写？示例中为'Fabric'）
                if 'Fabric' in item:
                    material = item['Fabric']
                if 'Color' in item:
                    color = item['Color']
    except (ValueError, SyntaxError, TypeError):
        # 解析失败时返回None
        pass
    return material, color

def preprocess_data(input_file, output_file):
    # 读取CSV文件
    df = pd.read_excel(input_file, engine='openpyxl')
    print(f"原始数据行数: {len(df)}")

    # 1. 去除不需要的字段
    drop_cols = ['seller', 'url', '_id', 'crawled_at', 'out_of_stock']
    # 确保这些列存在（不存在则忽略）
    existing_drop = [col for col in drop_cols if col in df.columns]
    df.drop(columns=existing_drop, inplace=True)
    print(f"已删除列: {existing_drop}")

    # 2. 去除images为空的数据
    # 先解析images列，判断是否为有效非空列表
    def is_images_empty(img_str):
        if pd.isna(img_str) or img_str == '':
            return True
        try:
            img_list = ast.literal_eval(img_str)
            if isinstance(img_list, list) and len(img_list) > 0:
                return False
            else:
                return True
        except:
            return True

    # 创建布尔掩码，保留非空images的行
    mask_images = ~df['images'].apply(is_images_empty)
    df = df[mask_images]
    print(f"去除images为空后行数: {len(df)}")

    # 3. 去除discount为空的数据
    df = df.dropna(subset=['discount'])
    # 进一步去除空字符串
    df = df[df['discount'].astype(str).str.strip() != '']
    print(f"去除discount为空后行数: {len(df)}")

    # 4. 交换discount和description字段
    if 'discount' in df.columns and 'description' in df.columns:
        df['discount'], df['description'] = df['description'], df['discount']
        print("已交换discount和description列")
    else:
        print("警告: discount或description列不存在，无法交换")

    # 5. 从product_details中提取材质和颜色
    df[['材质', '颜色']] = df['product_details'].apply(
        lambda x: pd.Series(extract_material_color(x))
    )
    print("已提取材质和颜色")

    # 6. 组合产品参数: category, sub_category, 材质, 颜色, brand
    # 若某些列缺失，则用空字符串替代
    def combine_params(row):
        parts = []
        for col in ['category', 'sub_category', '材质', '颜色', 'brand']:
            val = row.get(col, '')
            if pd.isna(val):
                val = ''
            parts.append(str(val))
        return ', '.join(parts)

    df['parameters'] = df.apply(combine_params, axis=1)
    print("已生成产品参数")

    # 7. 保存所需字段为单独的CSV文件
    # 产品id: pid, 产品名称: title, 产品描述: description, 产品参数: parameters
    output_cols = ['pid', 'title', 'description', 'parameters']
    # 确保列存在
    existing_output = [col for col in output_cols if col in df.columns]
    if len(existing_output) != len(output_cols):
        missing = set(output_cols) - set(existing_output)
        print(f"警告: 以下列不存在: {missing}")
    df_out = df[existing_output]
    df_out.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"预处理完成，输出文件: {output_file}，行数: {len(df_out)}")

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    input_file = project_root / "data" / "raw" / "input.xlsx"  # 原始数据文件
    output_file = project_root / "data" / "processed" / "processed.csv"  # 输出文件

    preprocess_data(input_file, output_file)
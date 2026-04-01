"""
环境依赖测试脚本
用于检查关键库是否正确安装，并显示其版本和所在路径
"""
import os
import sys

# 指定 Anaconda 环境的 Library\bin 路径（请根据你的实际路径修改）
conda_lib_bin = r'D:\anaconda\envs\Aproject\Library\bin'

if os.path.exists(conda_lib_bin):
    # Python 3.8+ 的方法
    os.add_dll_directory(conda_lib_bin)
else:
    print(f"警告：路径 {conda_lib_bin} 不存在，请检查。")

import importlib
from pathlib import Path

# ========================= 配置区 =========================
# 需要测试的库及其说明
# 格式：'库名': {'required': True/False, 'import_name': 实际导入名（可选，默认与库名相同）}
# required=True 表示该库必须存在，否则会报错并退出
libraries_to_test = {
    # 基础科学计算
    'numpy': {'required': True},
    'pandas': {'required': True},

    # Excel 处理
    'openpyxl': {'required': True},  # 若不需要可改为 False
    'tqdm': {'required': False},  # 若不需要可改为 False

    # 机器学习 / 深度学习
    'sklearn': {'required': False, 'import_name': 'sklearn'},  # 注意：通常导入 sklearn 而不是 scikit-learn
    'torch': {'required': False},
    'transformers': {'required': False},
    'seqeval': {'required': False},
    'joblib':{'required': False},
    'shap':{'required': True},
    'scikit-learn':{'required': True},

    # 其他常用标准库（标准库无需测试，但为了完整性可以保留）
    'pathlib': {'required': False, 'is_std': True},
    'ast': {'required': False, 'is_std': True},
    'warnings': {'required': False, 'is_std': True},
    # 说明：原脚本使用 'xz' 作为必装项，可能导致环境检查误失败；
    # 这里改为标准库 'lzma'（对应 xz 压缩格式的实现）。
    'lzma': {'required': True, 'is_std': True},
}


# ========================= 测试函数 =========================
def test_library(name, required=True, import_name=None, is_std=False):
    """测试单个库的导入并打印信息"""
    if is_std:
        # 标准库直接导入，不打印版本
        try:
            lib = importlib.import_module(name)
            print(f"[OK] {name:15s} (标准库)  路径: {lib.__file__ if hasattr(lib, '__file__') else '内置'}")
            return True
        except ImportError:
            print(f"[FAIL] {name:15s} (标准库)  未找到")
            return False

    # 第三方库：尝试导入并获取版本
    try:
        # 确定实际导入名
        module_name = import_name if import_name else name
        lib = importlib.import_module(module_name)

        # 获取版本（不同库的属性可能不同）
        version = getattr(lib, '__version__', '未知')
        file_path = getattr(lib, '__file__', '未知')

        # 检查是否来自 conda 环境（非用户目录）
        conda_env_path = Path(sys.prefix) / 'Lib' / 'site-packages'
        is_in_conda = str(conda_env_path) in file_path if file_path != '未知' else False
        location = file_path if file_path != '未知' else '内置'
        location_info = f"{location} (OK Conda环境)" if is_in_conda else location

        print(f"[OK] {name:15s} 版本: {version:<12} 路径: {location_info}")
        return True

    except ImportError:
        if required:
            print(f"[FAIL] {name:15s} 未安装（必要库，脚本将退出）")
            sys.exit(1)
        else:
            print(f"[WARN] {name:15s} 未安装（可选，跳过）")
            return False
    except Exception as e:
        print(f"[WARN] {name:15s} 导入时出错: {e}")
        return False


# ========================= 主程序 =========================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Python 环境依赖检查")
    print(f"  Python 解释器: {sys.executable}")
    print(f"  Python 版本: {sys.version}")
    print("=" * 60 + "\n")

    # 逐个测试库
    for lib_name, info in libraries_to_test.items():
        test_library(
            name=lib_name,
            required=info.get('required', False),
            import_name=info.get('import_name', None),
            is_std=info.get('is_std', False)
        )
    # 测试torch.cuda
    import torch
    cuda_available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_available}")
    if cuda_available:
        # CUDA 可用时才获取设备信息，避免无 CUDA 环境直接崩溃
        print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
        print(f"torch.version.cuda: {torch.version.cuda}")
    else:
        print("CUDA不可用（跳过 get_device_name / CUDA 版本）")

    print("\n" + "=" * 60)
    print(" 测试完成。")
    print("=" * 60)
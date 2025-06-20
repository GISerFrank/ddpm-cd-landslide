# data/validators.py

import torch
import numpy as np

class DataValidator:
    """
    数据验证器 - 负责验证标签和物理数据。
    采用单例模式，以确保验证信息只在第一次加载时打印一次。
    """
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.is_label_normalized = None
            self.label_validation_done = False
            self.physical_data_validated = False
            DataValidator._initialized = True
    
    def validate_and_fix_labels(self, data_dict: dict, phase: str = "train"):
        """
        验证并修复标签数据。
        'data_dict' 是一个包含键 'L' (PIL Image) 的字典。
        此方法会直接修改 data_dict 中的 'L'。
        """
        if 'L' not in data_dict or self.label_validation_done:
            return

        label_pil = data_dict['L']
        labels = torch.from_numpy(np.array(label_pil)) # 将PIL转换为Tensor进行分析
        
        unique_vals = torch.unique(labels)
        min_val, max_val = labels.min().item(), labels.max().item()
        
        print(f"\n🔍 [{phase}] 标签验证（仅显示一次）:")
        print(f"   形状: {labels.shape}, 数据类型: {labels.dtype}")
        print(f"   值范围: [{min_val}, {max_val}]")
        print(f"   唯一值: {unique_vals.tolist()}")
        
        self.is_label_normalized = (max_val <= 1.0 and min_val >= 0.0)
        
        if self.is_label_normalized:
            print("   🔧 检测到归一化标签，使用阈值二值化（阈值=0.5）")
            fixed_labels = (labels >= 0.5).long()
        else:
            print("   🔧 检测到标准标签，映射255→1")
            fixed_labels = labels.clone()
            if 255 in unique_vals:
                fixed_labels[labels == 255] = 1
            fixed_labels = torch.clamp(fixed_labels, 0, 1).long()
        
        final_unique = torch.unique(fixed_labels)
        print(f"   ✅ 修复完成: 唯一值{final_unique.tolist()}")
        
        # 将修复后的Tensor转换回PIL Image，以便后续的数据增强步骤可以统一处理
        # 注意：这里我们假设后续步骤会再转为Tensor。如果后续直接用Tensor，则无需转回PIL。
        # data_dict['L'] = Image.fromarray(fixed_labels.numpy().astype(np.uint8) * 255) # 或者直接传递Tensor
        data_dict['L_tensor'] = fixed_labels # 传递修复后的tensor
        
        self.label_validation_done = True
        print("   ✅ 标签验证设置完成，后续批次将快速处理\n")

    def validate_physical_data(self, data_dict: dict, phase: str = "train"):
        """
        验证物理数据。
        'data_dict' 是一个包含键 'physical_data' (Tensor) 的字典。
        """
        if 'physical_data' not in data_dict or data_dict['physical_data'] is None or self.physical_data_validated:
            return

        physical = data_dict['physical_data']
        print(f"\n🔍 [{phase}] 物理数据验证（仅显示一次）:")
        print(f"   形状: {physical.shape}, 数据类型: {physical.dtype}")
        print(f"   通道数: {physical.shape[0]}") # 假设通道在第一维
        
        channel_names = ['DEM', '坡度', '坡向', '地质类型', '植被覆盖']
        for i in range(min(physical.shape[0], len(channel_names))):
            channel_data = physical[i]
            print(f"   通道{i} ({channel_names[i]}): "
                  f"范围[{channel_data.min():.2f}, {channel_data.max():.2f}]")
        
        self.physical_data_validated = True
        print("   ✅ 物理数据验证完成\n")


# 创建一个全局单例供项目导入使用
data_validator = DataValidator()
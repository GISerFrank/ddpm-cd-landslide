import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
import cv2
from scipy import ndimage

# 原有评估指标保持不变，添加以下滑坡检测专用指标：

class LandslideMetrics:
    """
    滑坡检测专用评估指标
    """
    
    @staticmethod
    def calculate_comprehensive_metrics(pred, target, physical_data=None):
        """
        计算综合评估指标
        
        Args:
            pred: 预测结果 [B, 2, H, W] 或 [B, H, W]
            target: 真实标签 [B, H, W]
            physical_data: 物理数据 [B, num_layers, H, W] (可选)
        
        Returns:
            dict: 包含各种评估指标的字典
        """
        metrics = {}
        
        # 处理预测结果
        if pred.dim() == 4 and pred.shape[1] == 2:
            # Softmax + argmax for multi-class
            pred_binary = torch.argmax(pred, dim=1)
        elif pred.dim() == 3:
            # 已经是二值预测
            pred_binary = (pred > 0.5).float()
        else:
            raise ValueError(f"预测结果维度不正确: {pred.shape}")
        
        # 转换为numpy
        pred_np = pred_binary.cpu().numpy().astype(np.uint8)
        target_np = target.cpu().numpy().astype(np.uint8)
        
        # 基础指标
        metrics.update(LandslideMetrics._calculate_basic_metrics(pred_np, target_np))
        
        # 空间指标
        metrics.update(LandslideMetrics._calculate_spatial_metrics(pred_np, target_np))
        
        # 如果有物理数据，计算物理一致性指标
        if physical_data is not None:
            physical_np = physical_data.cpu().numpy()
            metrics.update(LandslideMetrics._calculate_physics_consistency(
                pred_np, target_np, physical_np))
        
        return metrics
    
    @staticmethod
    def _calculate_basic_metrics(pred, target):
        """计算基础分类指标"""
        metrics = {}
        
        # 展平数组用于计算
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        
        # 混淆矩阵
        tn, fp, fn, tp = confusion_matrix(target_flat, pred_flat, labels=[0, 1]).ravel()
        
        # 基础指标
        metrics['true_positive'] = int(tp)
        metrics['true_negative'] = int(tn)
        metrics['false_positive'] = int(fp)
        metrics['false_negative'] = int(fn)
        
        # 准确率相关
        metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # F1分数
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1_score'] = 2 * metrics['precision'] * metrics['recall'] / (
                metrics['precision'] + metrics['recall'])
        else:
            metrics['f1_score'] = 0
        
        # IoU
        metrics['iou'] = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        
        # 平衡准确率
        metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2
        
        return metrics
    
    @staticmethod
    def _calculate_spatial_metrics(pred, target):
        """计算空间相关指标"""
        metrics = {}
        
        batch_size = pred.shape[0]
        total_hausdorff = 0
        total_boundary_f1 = 0
        total_connectivity = 0
        valid_samples = 0
        
        for i in range(batch_size):
            pred_i = pred[i]
            target_i = target[i]
            
            # 检查是否有滑坡区域
            if np.sum(target_i) == 0 and np.sum(pred_i) == 0:
                # 都是非滑坡区域，跳过空间指标计算
                continue
            
            valid_samples += 1
            
            # Hausdorff距离
            hausdorff_dist = LandslideMetrics._calculate_hausdorff_distance(pred_i, target_i)
            total_hausdorff += hausdorff_dist
            
            # 边界F1分数
            boundary_f1 = LandslideMetrics._calculate_boundary_f1(pred_i, target_i)
            total_boundary_f1 += boundary_f1
            
            # 连通性指标
            connectivity = LandslideMetrics._calculate_connectivity_score(pred_i, target_i)
            total_connectivity += connectivity
        
        # 平均空间指标
        if valid_samples > 0:
            metrics['hausdorff_distance'] = total_hausdorff / valid_samples
            metrics['boundary_f1'] = total_boundary_f1 / valid_samples
            metrics['connectivity_score'] = total_connectivity / valid_samples
        else:
            metrics['hausdorff_distance'] = 0
            metrics['boundary_f1'] = 0
            metrics['connectivity_score'] = 0
        
        return metrics
    
    @staticmethod
    def _calculate_hausdorff_distance(pred, target):
        """计算Hausdorff距离"""
        try:
            from scipy.spatial.distance import directed_hausdorff
            
            # 获取边界点
            pred_points = np.column_stack(np.where(pred == 1))
            target_points = np.column_stack(np.where(target == 1))
            
            if len(pred_points) == 0 or len(target_points) == 0:
                return 100.0  # 如果一个为空，返回大值
            
            # 计算双向Hausdorff距离
            dist1 = directed_hausdorff(pred_points, target_points)[0]
            dist2 = directed_hausdorff(target_points, pred_points)[0]
            
            return max(dist1, dist2)
        
        except Exception:
            return 50.0  # 计算失败时返回中等值
    
    @staticmethod
    def _calculate_boundary_f1(pred, target):
        """计算边界F1分数"""
        try:
            # 计算边界
            pred_boundary = LandslideMetrics._get_boundary(pred)
            target_boundary = LandslideMetrics._get_boundary(target)
            
            # 在边界附近设置容差
            tolerance = 2
            pred_dilated = ndimage.binary_dilation(pred_boundary, 
                                                 structure=np.ones((tolerance*2+1, tolerance*2+1)))
            target_dilated = ndimage.binary_dilation(target_boundary,
                                                   structure=np.ones((tolerance*2+1, tolerance*2+1)))
            
            # 计算边界匹配
            tp = np.sum(pred_boundary & target_dilated)
            fp = np.sum(pred_boundary & ~target_dilated)
            fn = np.sum(target_boundary & ~pred_dilated)
            
            if tp + fp == 0:
                precision = 0
            else:
                precision = tp / (tp + fp)
            
            if tp + fn == 0:
                recall = 0
            else:
                recall = tp / (tp + fn)
            
            if precision + recall == 0:
                return 0
            else:
                return 2 * precision * recall / (precision + recall)
        
        except Exception:
            return 0
    
    @staticmethod
    def _get_boundary(mask):
        """获取边界"""
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
        boundary = mask.astype(np.uint8) - eroded
        return boundary.astype(bool)
    
    @staticmethod
    def _calculate_connectivity_score(pred, target):
        """计算连通性分数"""
        try:
            # 连通组件分析
            pred_labeled, pred_num = ndimage.label(pred)
            target_labeled, target_num = ndimage.label(target)
            
            if target_num == 0:
                return 1.0 if pred_num == 0 else 0.0
            
            # 计算预测和真实连通组件的匹配程度
            match_score = 0
            total_target_area = np.sum(target)
            
            for target_label in range(1, target_num + 1):
                target_component = (target_labeled == target_label)
                target_area = np.sum(target_component)
                
                # 找到与该真实组件重叠最大的预测组件
                best_overlap = 0
                for pred_label in range(1, pred_num + 1):
                    pred_component = (pred_labeled == pred_label)
                    overlap = np.sum(target_component & pred_component)
                    best_overlap = max(best_overlap, overlap)
                
                # 按面积加权的匹配分数
                match_score += (best_overlap / target_area) * (target_area / total_target_area)
            
            return match_score
        
        except Exception:
            return 0
    
    @staticmethod
    def _calculate_physics_consistency(pred, target, physical_data):
        """计算物理一致性指标"""
        metrics = {}
        
        batch_size = pred.shape[0]
        
        if physical_data.shape[1] >= 2:  # 至少有DEM和坡度数据
            slope_consistency = 0
            valid_samples = 0
            
            for i in range(batch_size):
                pred_i = pred[i]
                slope_i = physical_data[i, 1]  # 假设第二个通道是坡度
                
                if np.sum(pred_i) > 0:  # 如果有预测的滑坡
                    # 计算预测滑坡区域的平均坡度
                    pred_slope_mean = np.mean(slope_i[pred_i == 1])
                    
                    # 坡度合理性评分 (15-45度被认为是理想范围)
                    if 15 <= pred_slope_mean <= 45:
                        slope_score = 1.0
                    elif 5 <= pred_slope_mean < 15 or 45 < pred_slope_mean <= 70:
                        slope_score = 0.5
                    else:
                        slope_score = 0.0
                    
                    slope_consistency += slope_score
                    valid_samples += 1
            
            if valid_samples > 0:
                metrics['slope_consistency'] = slope_consistency / valid_samples
            else:
                metrics['slope_consistency'] = 1.0
        
        return metrics


def calculate_landslide_metrics(pred, target, physical_data=None):
    """
    计算滑坡检测的综合评估指标
    
    这是主要的接口函数，用于替换或补充原有的指标计算
    """
    return LandslideMetrics.calculate_comprehensive_metrics(pred, target, physical_data)


def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / \
        (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(
            math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # cv2.imwrite(img_path, img)

def save_feat(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, cv2.applyColorMap(cv2.resize(img, (256,256), interpolation=cv2.INTER_CUBIC), cv2.COLORMAP_JET))
    # cv2.imwrite(img_path, cv2.resize(img, (256,256), interpolation=cv2.INTER_))
    # cv2.imwrite(img_path, img)


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

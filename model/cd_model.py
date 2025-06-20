# import logging
# from collections import OrderedDict

# import torch
# import torch.nn as nn
# import os
# import model.networks as networks
# from .base_model import BaseModel
# from misc.metric_tools import ConfuseMatrixMeter
# from misc.torchutils import get_scheduler
# logger = logging.getLogger('base')


# class CD(BaseModel):
#     def __init__(self, opt):
#         super(CD, self).__init__(opt)
#         # define network and load pretrained models
#         self.netCD = self.set_device(networks.define_CD(opt))

#         # set loss and load resume state
#         self.loss_type = opt['model_cd']['loss_type']
#         if self.loss_type == 'ce':
#             self.loss_func =nn.CrossEntropyLoss().to(self.device)
#         else:
#             raise NotImplementedError()
        
#         if self.opt['phase'] == 'train':
#             self.netCD.train()
#             # find the parameters to optimize
#             optim_cd_params = list(self.netCD.parameters())

#             if opt['train']["optimizer"]["type"] == "adam":
#                 self.optCD = torch.optim.Adam(
#                     optim_cd_params, lr=opt['train']["optimizer"]["lr"])
#             elif opt['train']["optimizer"]["type"] == "adamw":
#                 self.optCD = torch.optim.AdamW(
#                     optim_cd_params, lr=opt['train']["optimizer"]["lr"])
#             else:
#                 raise NotImplementedError(
#                     'Optimizer [{:s}] not implemented'.format(opt['train']["optimizer"]["type"]))
#             self.log_dict = OrderedDict()
            
#             #Define learning rate sheduler
#             self.exp_lr_scheduler_netCD = get_scheduler(optimizer=self.optCD, args=opt['train'])
#         else:
#             self.netCD.eval()
#             self.log_dict = OrderedDict()

#         self.load_network()
#         self.print_network()

#         self.running_metric = ConfuseMatrixMeter(n_class=opt['model_cd']['out_channels'])
#         self.len_train_dataloader = opt["len_train_dataloader"]
#         self.len_val_dataloader = opt["len_val_dataloader"]

#     # Feeding all data to the CD model
#     def feed_data(self, feats_A, feats_B, data):
#         self.feats_A = feats_A
#         self.feats_B = feats_B
#         self.data = self.set_device(data)

#     # Optimize the parameters of the CD model
#     def optimize_parameters(self):
#         self.optCD.zero_grad()
#         self.pred_cm = self.netCD(self.feats_A, self.feats_B)
#         l_cd = self.loss_func(self.pred_cm, self.data["L"].long())
#         l_cd.backward()
#         self.optCD.step()
#         self.log_dict['l_cd'] = l_cd.item()

#     # Testing on given data
#     def test(self):
#         self.netCD.eval()
#         with torch.no_grad():
#             if isinstance(self.netCD, nn.DataParallel):
#                 self.pred_cm = self.netCD.module.forward(self.feats_A, self.feats_B)
#             else:
#                 self.pred_cm = self.netCD(self.feats_A, self.feats_B)
#             l_cd = self.loss_func(self.pred_cm, self.data["L"].long())
#             self.log_dict['l_cd'] = l_cd.item()
#         self.netCD.train()

#     # Get current log
#     def get_current_log(self):
#         return self.log_dict

#     # Get current visuals
#     def get_current_visuals(self):
#         out_dict = OrderedDict()
#         out_dict['pred_cm'] = torch.argmax(self.pred_cm, dim=1, keepdim=False)
#         out_dict['gt_cm'] = self.data['L']
#         return out_dict

#     # Printing the CD network
#     def print_network(self):
#         s, n = self.get_network_description(self.netCD)
#         if isinstance(self.netCD, nn.DataParallel):
#             net_struc_str = '{} - {}'.format(self.netCD.__class__.__name__,
#                                              self.netCD.module.__class__.__name__)
#         else:
#             net_struc_str = '{}'.format(self.netCD.__class__.__name__)

#         logger.info(
#             'Change Detection Network structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
#         logger.info(s)

#     # Saving the network parameters
#     def save_network(self, epoch, is_best_model = False):
#         cd_gen_path = os.path.join(
#             self.opt['path']['checkpoint'], 'cd_model_E{}_gen.pth'.format(epoch))
#         cd_opt_path = os.path.join(
#             self.opt['path']['checkpoint'], 'cd_model_E{}_opt.pth'.format(epoch))
        
#         if is_best_model:
#             best_cd_gen_path = os.path.join(
#                 self.opt['path']['checkpoint'], 'best_cd_model_gen.pth'.format(epoch))
#             best_cd_opt_path = os.path.join(
#                 self.opt['path']['checkpoint'], 'best_cd_model_opt.pth'.format(epoch))

#         # Save CD model pareamters
#         network = self.netCD
#         if isinstance(self.netCD, nn.DataParallel):
#             network = network.module
#         state_dict = network.state_dict()
#         for key, param in state_dict.items():
#             state_dict[key] = param.cpu()
#         torch.save(state_dict, cd_gen_path)
#         if is_best_model:
#             torch.save(state_dict, best_cd_gen_path)


#         # Save CD optimizer paramers
#         opt_state = {'epoch': epoch,
#                      'scheduler': None, 
#                      'optimizer': None}
#         opt_state['optimizer'] = self.optCD.state_dict()
#         torch.save(opt_state, cd_opt_path)
#         if is_best_model:
#             torch.save(opt_state, best_cd_opt_path)

#         # Print info
#         logger.info(
#             'Saved current CD model in [{:s}] ...'.format(cd_gen_path))
#         if is_best_model:
#             logger.info(
#             'Saved best CD model in [{:s}] ...'.format(best_cd_gen_path))

#     # Loading pre-trained CD network
#     def load_network(self):
#         load_path = self.opt['path_cd']['resume_state']
#         if load_path is not None:
#             logger.info(
#                 'Loading pretrained model for CD model [{:s}] ...'.format(load_path))
#             gen_path = '{}_gen.pth'.format(load_path)
#             opt_path = '{}_opt.pth'.format(load_path)
            
#             # change detection model
#             network = self.netCD
#             if isinstance(self.netCD, nn.DataParallel):
#                 network = network.module
#             network.load_state_dict(torch.load(
#                 gen_path), strict=True)
            
#             if self.opt['phase'] == 'train':
#                 opt = torch.load(opt_path)
#                 self.optCD.load_state_dict(opt['optimizer'])
#                 self.begin_step = opt['iter']
#                 self.begin_epoch = opt['epoch']
    
#     # Functions related to computing performance metrics for CD
#     def _update_metric(self):
#         """
#         update metric
#         """
#         G_pred = self.pred_cm.detach()
#         G_pred = torch.argmax(G_pred, dim=1)

#         current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=self.data['L'].detach().cpu().numpy())
#         return current_score
    
#     # Collecting status of the current running batch
#     def _collect_running_batch_states(self):
#         self.running_acc = self._update_metric()
#         self.log_dict['running_acc'] = self.running_acc.item()

#     # Collect the status of the epoch
#     def _collect_epoch_states(self):
#         scores = self.running_metric.get_scores()
#         self.epoch_acc = scores['mf1']
#         self.log_dict['epoch_acc'] = self.epoch_acc.item()

#         for k, v in scores.items():
#             self.log_dict[k] = v
#             #message += '%s: %.5f ' % (k, v)

#     # Rest all the performance metrics
#     def _clear_cache(self):
#         self.running_metric.clear()

#     # Finctions related to learning rate sheduler
#     def _update_lr_schedulers(self):
#         self.exp_lr_scheduler_netCD.step()

        
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import os
import model.networks as networks
from .base_model import BaseModel
from misc.metric_tools import ConfuseMatrixMeter
from misc.torchutils import get_scheduler

# 导入物理约束损失函数
from .physical_loss import create_physics_loss

logger = logging.getLogger('base')

class DDPMCDModel(BaseModel):
    """
    DDPM变化检测模型，集成物理约束损失
    """
    def __init__(self, opt):
        super(DDPMCDModel, self).__init__(opt)
        
        # 初始化必要的属性（BaseModel没有提供）
        self.optimizers = []
        self.schedulers = []
        self.is_train = opt['phase'] == 'train'
        
        # 定义网络
        self.netCD = self.set_device(networks.define_CD(opt))

        # **重要修改：无论训练还是测试都需要设置损失函数**
        self.setup_loss_functions(opt)
        
        # 设置训练状态
        if self.is_train:
            self.netCD.train()
            
            # 创建优化器
            train_opt = opt['train']
            optim_params = list(self.netCD.parameters())
            
            if train_opt['optimizer']['type'] == 'adam':
                self.optimizer = torch.optim.Adam(
                    optim_params, 
                    lr=train_opt['optimizer']['lr'],
                    weight_decay=train_opt['optimizer'].get('weight_decay', 0)
                )
            elif train_opt['optimizer']['type'] == 'adamw':
                self.optimizer = torch.optim.AdamW(
                    optim_params,
                    lr=train_opt['optimizer']['lr'], 
                    weight_decay=train_opt['optimizer'].get('weight_decay', 1e-2)
                )
            else:
                raise NotImplementedError(f"优化器 {train_opt['optimizer']['type']} 未实现")
            
            # 为了兼容性，同时保存为optCD
            self.optCD = self.optimizer
            self.optimizers.append(self.optimizer)
            
            # 创建学习率调度器
            self.setup_schedulers()
            
            # 创建损失函数
            self.setup_loss_functions(opt)
            
            # 训练指标
            self.log_dict = OrderedDict()
        else:
            # 测试模式
            self.netCD.eval()
            self.log_dict = OrderedDict()
        
        # 初始化性能指标跟踪器
        self.running_metric = ConfuseMatrixMeter(n_class=opt['model_cd']['out_channels'])
        
        # 数据集长度信息
        if 'len_train_dataloader' in opt:
            self.len_train_dataloader = opt["len_train_dataloader"]
        if 'len_val_dataloader' in opt:
            self.len_val_dataloader = opt["len_val_dataloader"]
        
        # 加载预训练模型
        self.load_network()
        
        # 打印网络信息
        self.print_network()
        
    def setup_loss_functions(self, opt):
        """设置损失函数"""
        loss_type = opt['model_cd'].get('loss_type', 'ce')
        
        if loss_type == 'physics_constrained':
            # 使用物理约束损失
            physics_config = opt['model_cd'].get('physics_loss', {})
            self.criterion = create_physics_loss(physics_config)
            self.criterion = self.criterion.to(self.device)
            print("🔧 使用物理约束损失函数")
            
        elif loss_type == 'ce':
            # 标准交叉熵损失
            self.criterion = nn.CrossEntropyLoss().to(self.device)
            print("🔧 使用交叉熵损失函数")
            
        elif loss_type == 'bce':
            # 二元交叉熵损失
            self.criterion = nn.BCEWithLogitsLoss().to(self.device)
            print("🔧 使用二元交叉熵损失函数")
            
        elif loss_type == 'focal':
            # Focal损失
            self.criterion = self.focal_loss
            print("🔧 使用Focal损失函数")
            
        else:
            raise NotImplementedError(f"损失类型 {loss_type} 未实现")
        
        # 为了兼容性，同时保存为loss_func
        self.loss_func = self.criterion
    
    def setup_schedulers(self):
        """设置学习率调度器"""
        if self.is_train and 'scheduler' in self.opt['train']:
            self.exp_lr_scheduler_netCD = get_scheduler(
                optimizer=self.optimizer, 
                args=self.opt['train']
            )
            if self.exp_lr_scheduler_netCD:
                self.schedulers.append(self.exp_lr_scheduler_netCD)
    
    def focal_loss(self, pred, target, alpha=1, gamma=2):
        """Focal Loss实现"""
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1-pt)**gamma * ce_loss
        return focal_loss.mean()
    
    # 兼容旧接口的feed_data
    def feed_data(self, *args):
        """输入数据 - 兼容新旧接口"""
        if len(args) == 1:
            # 新接口：feed_data(data)
            data = args[0]
            self.data = self.set_device(data)
            self.image_A = self.data.get('A')
            self.image_B = self.data.get('B')
            self.label = self.data.get('label', self.data.get('L'))  # 兼容'L'和'label'
            
            # 物理数据（可选）
            self.physical_data = self.data.get('physical_data', None)
        else:
            # 旧接口：feed_data(feats_A, feats_B, data)
            self.feats_A = args[0]
            self.feats_B = args[1]
            self.data = self.set_device(args[2])
            self.label = self.data.get('label', self.data.get('L'))
            self.physical_data = self.data.get('physical_data', None)
    
    def forward_cd(self, features_A, features_B):
        """变化检测前向传播"""
        change_map = self.netCD(features_A, features_B)
        return change_map
    
    # 兼容旧接口的optimize_parameters
    def optimize_parameters(self, *args, **kwargs):
        """优化参数 - 兼容新旧接口"""
        if len(args) >= 2:
            # 新接口：optimize_parameters(features_A, features_B, current_epoch=None)
            features_A = args[0]
            features_B = args[1]
            current_epoch = kwargs.get('current_epoch', None)
        else:
            # 旧接口：optimize_parameters() - 使用保存的特征
            if hasattr(self, 'feats_A') and hasattr(self, 'feats_B'):
                features_A = self.feats_A
                features_B = self.feats_B
            elif hasattr(self, '_temp_features_A') and hasattr(self, '_temp_features_B'):
                features_A = self._temp_features_A
                features_B = self._temp_features_B
            else:
                raise ValueError("没有可用的特征进行优化")
            current_epoch = kwargs.get('current_epoch', None)
        
        self.optimizer.zero_grad()
        
        # 前向传播
        self.pred_cm = self.forward_cd(features_A, features_B)
        self.change_prediction = self.pred_cm  # 为了兼容性
        
        # 计算损失
        if hasattr(self.criterion, 'forward') and 'physical_data' in self.criterion.forward.__code__.co_varnames:
            # 物理约束损失函数
            self.loss_cd = self.criterion(
                self.pred_cm, 
                self.label.long(),
                physical_data=self.physical_data,
                epoch=current_epoch
            )
        else:
            # 标准损失函数
            self.loss_cd = self.criterion(self.pred_cm, self.label.long())
        
        # 反向传播
        self.loss_cd.backward()
        self.optimizer.step()
        
        # 记录损失
        self.log_dict['l_cd'] = self.loss_cd.item()
    
    # 兼容旧接口的test
    def test(self, *args):
        """测试模式 - 兼容新旧接口"""
        self.netCD.eval()
        with torch.no_grad():
            if len(args) >= 2:
                # 新接口：test(features_A, features_B)
                features_A = args[0]
                features_B = args[1]
            else:
                # 旧接口：test() - 使用保存的特征
                if hasattr(self, 'feats_A') and hasattr(self, 'feats_B'):
                    features_A = self.feats_A
                    features_B = self.feats_B
                elif hasattr(self, '_temp_features_A') and hasattr(self, '_temp_features_B'):
                    features_A = self._temp_features_A
                    features_B = self._temp_features_B
                else:
                    raise ValueError("没有可用的特征进行测试")
            
            self.pred_cm = self.forward_cd(features_A, features_B)
            self.change_prediction = self.pred_cm
            
            # 计算损失
            # 修复：优先使用self.criterion，兼容self.loss_func
            if hasattr(self, 'criterion') and self.criterion is not None:
                l_cd = self.criterion(self.pred_cm, self.label.long())
            elif hasattr(self, 'loss_func') and self.loss_func is not None:
                l_cd = self.loss_func(self.pred_cm, self.label.long())
            self.log_dict['l_cd'] = l_cd.item()
        
        if self.is_train:
            self.netCD.train()
    
    def get_current_log(self):
        """获取当前日志"""
        return self.log_dict
    
    def get_current_visuals(self):
        """获取当前可视化结果"""
        out_dict = OrderedDict()
        
        # 预测结果
        if hasattr(self, 'pred_cm'):
            pred_cm = torch.argmax(self.pred_cm, dim=1, keepdim=False)
            out_dict['pred_cm'] = pred_cm.detach().float().cpu()
        
        # 真实标签
        if hasattr(self, 'label'):
            out_dict['gt_cm'] = self.label.detach().float().cpu()
        elif hasattr(self, 'data') and 'L' in self.data:
            out_dict['gt_cm'] = self.data['L'].detach().float().cpu()
        
        return out_dict
    
    def print_network(self):
        """打印网络结构"""
        s, n = self.get_network_description(self.netCD)
        if isinstance(self.netCD, nn.DataParallel):
            net_struc_str = f'{self.netCD.__class__.__name__} - {self.netCD.module.__class__.__name__}'
        else:
            net_struc_str = f'{self.netCD.__class__.__name__}'
        
        logger.info(f'Change Detection Network structure: {net_struc_str}, with parameters: {n:,d}')
        logger.info(s)
    
    def save_network(self, epoch, is_best_model=False):
        """保存网络参数 - 兼容旧接口"""
        cd_gen_path = os.path.join(
            self.opt['path']['checkpoint'], f'cd_model_E{epoch}_gen.pth')
        cd_opt_path = os.path.join(
            self.opt['path']['checkpoint'], f'cd_model_E{epoch}_opt.pth')
        
        if is_best_model:
            best_cd_gen_path = os.path.join(
                self.opt['path']['checkpoint'], 'best_cd_model_gen.pth')
            best_cd_opt_path = os.path.join(
                self.opt['path']['checkpoint'], 'best_cd_model_opt.pth')

        # 保存CD模型参数
        network = self.netCD
        if isinstance(self.netCD, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, cd_gen_path)
        if is_best_model:
            torch.save(state_dict, best_cd_gen_path)

        # 保存CD优化器参数
        opt_state = {'epoch': epoch,
                     'scheduler': None, 
                     'optimizer': None}
        opt_state['optimizer'] = self.optimizer.state_dict()
        torch.save(opt_state, cd_opt_path)
        if is_best_model:
            torch.save(opt_state, best_cd_opt_path)

        # 打印信息
        logger.info(f'Saved current CD model in [{cd_gen_path}] ...')
        if is_best_model:
            logger.info(f'Saved best CD model in [{best_cd_gen_path}] ...')
    
    def load_network(self):
        """加载预训练的CD网络"""
        load_path = self.opt.get('path_cd', {}).get('resume_state')
        if load_path is not None:
            logger.info(f'Loading pretrained model for CD model [{load_path}] ...')
            gen_path = f'{load_path}_gen.pth'
            opt_path = f'{load_path}_opt.pth'
            
            # 加载变化检测模型
            if os.path.exists(gen_path):
                network = self.netCD
                if isinstance(self.netCD, nn.DataParallel):
                    network = network.module
                network.load_state_dict(torch.load(gen_path), strict=True)
                
                if self.is_train and os.path.exists(opt_path):
                    opt = torch.load(opt_path)
                    self.optimizer.load_state_dict(opt['optimizer'])
                    if 'epoch' in opt:
                        self.begin_epoch = opt['epoch']
                    if 'iter' in opt:
                        self.begin_step = opt['iter']
    
    # CD特有的方法
    def _update_metric(self):
        """更新指标"""
        G_pred = self.pred_cm.detach()
        G_pred = torch.argmax(G_pred, dim=1)
        
        current_score = self.running_metric.update_cm(
            pr=G_pred.cpu().numpy(), 
            gt=self.label.detach().cpu().numpy()
        )
        return current_score
    
    def _collect_running_batch_states(self):
        """收集当前批次的运行状态"""
        self.running_acc = self._update_metric()
        self.log_dict['running_acc'] = self.running_acc.item()
    
    def _collect_epoch_states(self):
        """收集epoch状态"""
        scores = self.running_metric.get_scores()
        self.epoch_acc = scores['mf1']
        self.log_dict['epoch_acc'] = self.epoch_acc.item()
        
        for k, v in scores.items():
            self.log_dict[k] = v
    
    def _clear_cache(self):
        """清除缓存"""
        self.running_metric.clear()
    
    def _update_lr_schedulers(self):
        """更新学习率调度器"""
        if hasattr(self, 'exp_lr_scheduler_netCD') and self.exp_lr_scheduler_netCD:
            self.exp_lr_scheduler_netCD.step()
        elif hasattr(self, 'schedulers'):
            for scheduler in self.schedulers:
                if scheduler:
                    scheduler.step()
    
    def get_physics_constraint_details(self):
        """获取物理约束损失的详细信息（用于分析）"""
        if hasattr(self.criterion, 'get_constraint_details'):
            return self.criterion.get_constraint_details(
                self.pred_cm if hasattr(self, 'pred_cm') else self.change_prediction, 
                self.label, 
                self.physical_data
            )
        return {}


# 为了向后兼容
CD = DDPMCDModel


def create_CD_model(opt):
    """创建变化检测模型"""
    model = DDPMCDModel(opt)
    logger.info(f'CD Model [{model.__class__.__name__}] is created.')
    return model
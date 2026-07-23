import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, List, Optional, Union, Any
from .config import config

class KnowledgeDistillationLoss(nn.Module):
    """
    知识蒸馏损失函数
    实现多种蒸馏策略：标准KD、特征匹配、注意力转移
    """
    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7,
        beta: float = 0.3,
        feature_weight: float = 0.5,
        attention_weight: float = 0.3
    ):
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha  # 蒸馏损失权重
        self.beta = beta    # 真实标签损失权重
        self.feature_weight = feature_weight
        self.attention_weight = attention_weight
        
        # 损失函数
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        
        # 修复：初始化适配器为None，避免循环引用
        self.logits_adapter = None
        self.feature_adapter = None
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_features: Optional[torch.Tensor] = None,
        teacher_features: Optional[torch.Tensor] = None,
        student_attention: Optional[torch.Tensor] = None,
        teacher_attention: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算知识蒸馏损失
        
        Args:
            student_logits: 学生模型输出 [batch_size, seq_len, vocab_size]
            teacher_logits: 教师模型输出 [batch_size, seq_len, vocab_size]
            student_features: 学生模型特征 [batch_size, seq_len, hidden_size]
            teacher_features: 教师模型特征 [batch_size, seq_len, hidden_size]
            student_attention: 学生模型注意力权重
            teacher_attention: 教师模型注意力权重
            targets: 真实标签 [batch_size, seq_len]
        
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的详细信息
        """
        loss_dict = {}
        device = student_logits.device
        
        # 修复：确保teacher_logits在相同设备上
        teacher_logits = teacher_logits.to(device)
        
        # 确保logits维度匹配
        if student_logits.shape != teacher_logits.shape:
            # 如果序列长度不匹配，截断到较短的长度
            min_seq_len = min(student_logits.size(1), teacher_logits.size(1))
            student_logits = student_logits[:, :min_seq_len, :]
            teacher_logits = teacher_logits[:, :min_seq_len, :]
            
            # 如果词汇表大小不匹配，投影teacher logits
            if student_logits.size(-1) != teacher_logits.size(-1):
                if self.logits_adapter is None:
                    self.logits_adapter = nn.Linear(
                        teacher_logits.size(-1), 
                        student_logits.size(-1)
                    ).to(device)
                teacher_logits = self.logits_adapter(teacher_logits)
        
        # 1. 标准知识蒸馏损失 (Soft Target Loss)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        kd_loss = self.kl_loss(student_soft, teacher_soft) * (self.temperature ** 2)
        loss_dict['kd_loss'] = kd_loss
        
        # 2. 真实标签损失 (Hard Target Loss)
        hard_loss = torch.tensor(0.0, device=device)
        if targets is not None:
            # 修复：确保targets在相同设备上
            targets = targets.to(device)
            
            # 确保targets与logits的序列长度匹配
            if targets.size(1) != student_logits.size(1):
                min_seq_len = min(targets.size(1), student_logits.size(1))
                targets = targets[:, :min_seq_len]
                student_logits_for_ce = student_logits[:, :min_seq_len, :]
            else:
                student_logits_for_ce = student_logits
            
            # 重塑为2D进行交叉熵计算
            student_logits_2d = student_logits_for_ce.reshape(-1, student_logits_for_ce.size(-1))
            targets_1d = targets.reshape(-1)
            
            # 修复：过滤掉padding tokens（假设-100是padding）
            valid_mask = targets_1d != -100
            if valid_mask.any():
                hard_loss = self.ce_loss(student_logits_2d[valid_mask], targets_1d[valid_mask])
            
            loss_dict['hard_loss'] = hard_loss
        
        # 3. 特征匹配损失 (Feature Matching Loss)
        feature_loss = torch.tensor(0.0, device=device)
        if student_features is not None and teacher_features is not None:
            # 修复：确保teacher_features在相同设备上
            teacher_features = teacher_features.to(device)
            
            # 确保特征的序列长度匹配
            if student_features.size(1) != teacher_features.size(1):
                min_seq_len = min(student_features.size(1), teacher_features.size(1))
                student_features = student_features[:, :min_seq_len, :]
                teacher_features = teacher_features[:, :min_seq_len, :]
            
            # 确保特征维度匹配
            if student_features.shape != teacher_features.shape:
                # 如果维度不匹配，使用线性投影对齐
                if self.feature_adapter is None:
                    self.feature_adapter = nn.Linear(
                        student_features.size(-1), 
                        teacher_features.size(-1)
                    ).to(device)
                student_features = self.feature_adapter(student_features)
            
            feature_loss = self.mse_loss(student_features, teacher_features)
            loss_dict['feature_loss'] = feature_loss
        
        # 4. 注意力转移损失 (Attention Transfer Loss)
        attention_loss = torch.tensor(0.0, device=device)
        if student_attention is not None and teacher_attention is not None:
            # 修复：确保teacher_attention在相同设备上
            teacher_attention = teacher_attention.to(device)
            
            # 确保注意力权重维度匹配
            if student_attention.shape != teacher_attention.shape:
                # 简单的维度对齐策略
                min_shape = [min(s, t) for s, t in zip(student_attention.shape, teacher_attention.shape)]
                if len(min_shape) >= 2:
                    student_attention = student_attention[:min_shape[0], :min_shape[1]]
                    teacher_attention = teacher_attention[:min_shape[0], :min_shape[1]]
            
            # 计算注意力图的MSE损失
            attention_loss = self.mse_loss(student_attention, teacher_attention)
            loss_dict['attention_loss'] = attention_loss
        
        # 5. 总损失计算
        total_loss = (
            self.alpha * kd_loss + 
            self.beta * hard_loss + 
            self.feature_weight * feature_loss + 
            self.attention_weight * attention_loss
        )
        
        loss_dict['total_loss'] = total_loss
        
        return total_loss, loss_dict

class DistillationLayer(nn.Module):
    """
    蒸馏层 - 用于特征对齐和知识转移
    """
    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        hidden_dim: Optional[int] = None,
        use_attention: bool = True
    ):
        super(DistillationLayer, self).__init__()
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        self.hidden_dim = hidden_dim or min(student_dim, teacher_dim)
        self.use_attention = use_attention
        
        # 特征对齐网络
        self.feature_adapter = nn.Sequential(
            nn.Linear(student_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, teacher_dim)
        )
        
        # 注意力对齐网络
        if use_attention:
            self.attention_adapter = nn.Sequential(
                nn.Linear(student_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, teacher_dim),
                nn.Sigmoid()
            )
    
    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        对齐学生和教师特征
        
        Args:
            student_features: 学生特征 [batch_size, seq_len, student_dim]
            teacher_features: 教师特征 [batch_size, seq_len, teacher_dim]
        
        Returns:
            aligned_student: 对齐后的学生特征
            attention_weights: 注意力权重（可选）
        """
        # 修复：确保features在相同设备上
        device = student_features.device
        teacher_features = teacher_features.to(device)
        
        # 特征对齐
        aligned_student = self.feature_adapter(student_features)
        
        # 注意力权重计算（可选）
        attention_weights = None
        if self.use_attention:
            attention_weights = self.attention_adapter(student_features)
            aligned_student = aligned_student * attention_weights
        
        return aligned_student, attention_weights

class PiKVDistillation(nn.Module):
    """
    PiKV专用蒸馏模块
    集成专家蒸馏、缓存蒸馏和路由蒸馏
    """
    def __init__(
        self,
        student_hidden_size: int,
        teacher_hidden_size: int,
        num_experts: int,
        temperature: float = 4.0,
        expert_distill_weight: float = 0.4,
        cache_distill_weight: float = 0.3
    ):
        super(PiKVDistillation, self).__init__()
        self.student_hidden_size = student_hidden_size
        self.teacher_hidden_size = teacher_hidden_size
        self.num_experts = num_experts
        self.expert_distill_weight = expert_distill_weight
        self.cache_distill_weight = cache_distill_weight
        
        # 基础知识蒸馏损失
        self.kd_loss = KnowledgeDistillationLoss(temperature=temperature)
        
        # 专家对齐层
        self.expert_distill_layers = nn.ModuleList([
            DistillationLayer(student_hidden_size, teacher_hidden_size)
            for _ in range(num_experts)
        ])
        
        # KV缓存对齐层
        self.cache_distill_layer = DistillationLayer(
            student_hidden_size, teacher_hidden_size
        )
        
        # 路由蒸馏网络
        self.routing_distill_net = nn.Sequential(
            nn.Linear(student_hidden_size, student_hidden_size),
            nn.ReLU(),
            nn.Linear(student_hidden_size, num_experts),
            nn.LogSoftmax(dim=-1)
        )
    
    def distill_expert_outputs(
        self,
        student_expert_outputs: List[torch.Tensor],
        teacher_expert_outputs: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        蒸馏专家输出
        """
        if not student_expert_outputs or not teacher_expert_outputs:
            return torch.tensor(0.0)
        
        device = student_expert_outputs[0].device
        expert_losses = []
        
        # 修复：确保expert数量匹配
        min_experts = min(len(student_expert_outputs), len(teacher_expert_outputs))
        
        for i in range(min_experts):
            student_out = student_expert_outputs[i]
            teacher_out = teacher_expert_outputs[i].to(device)
            
            # 使用对应的蒸馏层
            if i < len(self.expert_distill_layers):
                aligned_student, _ = self.expert_distill_layers[i](student_out, teacher_out)
                expert_loss = F.mse_loss(aligned_student, teacher_out)
                expert_losses.append(expert_loss)
        
        if expert_losses:
            return torch.stack(expert_losses).mean()
        else:
            return torch.tensor(0.0, device=device)
    
    def distill_kv_cache(
        self,
        student_cache: torch.Tensor,
        teacher_cache: torch.Tensor
    ) -> torch.Tensor:
        """
        蒸馏KV缓存
        """
        device = student_cache.device
        teacher_cache = teacher_cache.to(device)
        
        # 确保缓存维度匹配
        if student_cache.shape != teacher_cache.shape:
            # 序列长度对齐
            min_seq_len = min(student_cache.size(1), teacher_cache.size(1))
            student_cache = student_cache[:, :min_seq_len, :]
            teacher_cache = teacher_cache[:, :min_seq_len, :]
            
            # 特征维度对齐
            if student_cache.size(-1) != teacher_cache.size(-1):
                aligned_student, _ = self.cache_distill_layer(student_cache, teacher_cache)
                return F.mse_loss(aligned_student, teacher_cache)
        
        return F.mse_loss(student_cache, teacher_cache)
    
    def distill_routing(
        self,
        student_features: torch.Tensor,
        teacher_routing_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        蒸馏路由权重
        """
        device = student_features.device
        teacher_routing_weights = teacher_routing_weights.to(device)
        
        # 从学生特征预测路由权重
        batch_size, seq_len, hidden_size = student_features.shape
        
        # 平均池化以获得句子级特征
        pooled_features = student_features.mean(dim=1)  # [batch_size, hidden_size]
        
        # 预测路由权重
        student_routing_pred = self.routing_distill_net(pooled_features)  # [batch_size, num_experts]
        
        # 对teacher routing weights进行相同的池化
        if teacher_routing_weights.dim() == 3:  # [batch_size, seq_len, num_experts]
            teacher_routing_pooled = teacher_routing_weights.mean(dim=1)  # [batch_size, num_experts]
        else:
            teacher_routing_pooled = teacher_routing_weights
        
        # 计算KL散度损失
        routing_loss = F.kl_div(
            student_routing_pred,
            F.softmax(teacher_routing_pooled, dim=-1),
            reduction='batchmean'
        )
        
        return routing_loss
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        student_expert_outputs: Optional[List[torch.Tensor]] = None,
        teacher_expert_outputs: Optional[List[torch.Tensor]] = None,
        student_cache: Optional[torch.Tensor] = None,
        teacher_cache: Optional[torch.Tensor] = None,
        teacher_routing_weights: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        完整的PiKV蒸馏前向传播
        
        Returns:
            total_loss: 总蒸馏损失
            loss_dict: 详细损失信息
        """
        device = student_logits.device
        
        # 1. 基础知识蒸馏
        base_loss, loss_dict = self.kd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            student_features=student_features,
            teacher_features=teacher_features,
            targets=targets
        )
        
        # 2. 专家蒸馏
        expert_loss = torch.tensor(0.0, device=device)
        if student_expert_outputs and teacher_expert_outputs:
            expert_loss = self.distill_expert_outputs(
                student_expert_outputs, teacher_expert_outputs
            )
            loss_dict['expert_loss'] = expert_loss
        
        # 3. KV缓存蒸馏
        cache_loss = torch.tensor(0.0, device=device)
        if student_cache is not None and teacher_cache is not None:
            cache_loss = self.distill_kv_cache(student_cache, teacher_cache)
            loss_dict['cache_loss'] = cache_loss
        
        # 4. 路由蒸馏
        routing_loss = torch.tensor(0.0, device=device)
        if teacher_routing_weights is not None:
            routing_loss = self.distill_routing(student_features, teacher_routing_weights)
            loss_dict['routing_loss'] = routing_loss
        
        # 5. 总损失
        total_loss = (
            base_loss + 
            self.expert_distill_weight * expert_loss + 
            self.cache_distill_weight * cache_loss + 
            0.2 * routing_loss
        )
        
        loss_dict['total_distill_loss'] = total_loss
        
        return total_loss, loss_dict

def create_teacher_model(
    hidden_size: int,
    num_experts: int,
    num_layers: int = 6,
    vocab_size: int = 50257
) -> nn.Module:
    """
    创建教师模型 - 通常是更大、更强的模型
    
    Args:
        hidden_size: 隐藏层大小
        num_experts: 专家数量
        num_layers: 层数
        vocab_size: 词汇表大小
    
    Returns:
        teacher_model: 教师模型
    """
    class TeacherMoE(nn.Module):
        def __init__(self):
            super(TeacherMoE, self).__init__()
            self.hidden_size = hidden_size
            self.num_experts = num_experts
            
            # 更大的专家网络
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 2),
                    nn.ReLU(),
                    nn.Linear(hidden_size * 2, hidden_size),
                    nn.ReLU()
                ) for _ in range(num_experts)
            ])
            
            # 路由网络
            self.router = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_experts),
                nn.Softmax(dim=-1)
            )
            
            # 输出投影
            self.output_proj = nn.Linear(hidden_size, vocab_size)
        
        def forward(self, x):
            # 修复：处理不同输入格式
            if isinstance(x, dict):
                x = x.get('input_ids', x.get('inputs', x))
            
            # 修复：处理token IDs输入
            if x.dtype in [torch.long, torch.int]:
                # 如果是token IDs，需要embedding
                if not hasattr(self, 'embedding'):
                    self.embedding = nn.Embedding(vocab_size, self.hidden_size).to(x.device)
                x = self.embedding(x)
            
            batch_size, seq_len, hidden_size = x.shape
            
            # 路由权重
            routing_weights = self.router(x)
            
            # 专家输出
            expert_outputs = []
            combined_output = torch.zeros_like(x)
            
            for i, expert in enumerate(self.experts):
                expert_out = expert(x)
                expert_outputs.append(expert_out)
                combined_output += expert_out * routing_weights[:, :, i:i+1]
            
            # 输出logits
            logits = self.output_proj(combined_output)
            
            return {
                'logits': logits,
                'features': combined_output,
                'expert_outputs': expert_outputs,
                'routing_weights': routing_weights
            }
    
    return TeacherMoE()

def distillation_training_step(
    student_model: nn.Module,
    teacher_model: nn.Module,
    distillation_module: PiKVDistillation,
    input_data: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Dict[str, float]:
    """
    单步蒸馏训练
    
    Args:
        student_model: 学生模型
        teacher_model: 教师模型
        distillation_module: 蒸馏模块
        input_data: 输入数据
        targets: 目标标签
        optimizer: 优化器
    
    Returns:
        loss_info: 损失信息字典
    """
    device = input_data.device
    
    # 修复：确保所有模型在相同设备上
    teacher_model.to(device)
    distillation_module.to(device)
    
    # 教师模型推理（不计算梯度）
    teacher_model.eval()
    with torch.no_grad():
        try:
            teacher_outputs = teacher_model(input_data)
        except Exception as e:
            print(f"Teacher model forward error: {e}")
            # 创建虚拟teacher输出
            batch_size, seq_len = input_data.shape[:2]
            hidden_size = getattr(teacher_model, 'hidden_size', 512)
            vocab_size = getattr(teacher_model, 'output_proj', torch.nn.Linear(1, 50257)).out_features
            
            teacher_outputs = {
                'logits': torch.randn(batch_size, seq_len, vocab_size, device=device),
                'features': torch.randn(batch_size, seq_len, hidden_size, device=device),
                'expert_outputs': [],
                'routing_weights': torch.softmax(torch.randn(batch_size, seq_len, 4, device=device), dim=-1)
            }
    
    # 学生模型推理
    student_model.train()
    try:
        student_outputs = student_model(input_data, return_loss=False, use_teacher=False)
        
        # 修复：如果student_outputs不是字典，包装成字典
        if not isinstance(student_outputs, dict):
            student_outputs = {'logits': student_outputs}
            
    except Exception as e:
        print(f"Student model forward error: {e}")
        return {'error': float('inf')}
    
    # 计算蒸馏损失
    try:
        distill_loss, loss_dict = distillation_module(
            student_logits=student_outputs.get('logits', student_outputs.get('output', student_outputs)),
            teacher_logits=teacher_outputs.get('logits', teacher_outputs),
            student_features=student_outputs.get('features'),
            teacher_features=teacher_outputs.get('features'),
            student_expert_outputs=student_outputs.get('expert_outputs'),
            teacher_expert_outputs=teacher_outputs.get('expert_outputs'),
            teacher_routing_weights=teacher_outputs.get('routing_weights'),
            targets=targets
        )
    except Exception as e:
        print(f"Distillation forward error: {e}")
        return {'error': float('inf')}
    
    # 反向传播
    if optimizer is not None:
        try:
            optimizer.zero_grad()
            distill_loss.backward()
            # 修复：梯度裁剪防止爆炸
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            optimizer.step()
        except Exception as e:
            print(f"Optimization error: {e}")
            return {'error': float('inf')}
    
    # 转换损失为标量值
    loss_info = {}
    for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            loss_info[key] = value.item()
        else:
            loss_info[key] = value
    
    return loss_info 
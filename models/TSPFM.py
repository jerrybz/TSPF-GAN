import torch
from torch import nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class SpatialAttention(nn.Module):
    def __init__(self, channel, reduction=16, num_layers=3):
        super().__init__()
        self.sa = nn.Sequential()
        self.sa.add_module('conv_reduce1', nn.Conv2d(kernel_size=3,padding=1, in_channels=channel, out_channels=channel // reduction,groups=channel // reduction))
        self.sa.add_module('bn_reduce1', nn.BatchNorm2d(channel // reduction))
        self.sa.add_module('relu_reduce1', nn.PReLU())
        for i in range(num_layers):
            self.sa.add_module('conv_%d' % i, nn.Conv2d(
                kernel_size=3,
                in_channels=channel // reduction,
                out_channels=channel // reduction,
                padding=1,
                groups=channel // reduction,
            ))
            self.sa.add_module('bn_%d' % i, nn.BatchNorm2d(channel // reduction))
            self.sa.add_module('relu_%d' % i, nn.PReLU())
        self.sa.add_module('last_conv', nn.Conv2d(channel // reduction, 1, kernel_size=3,padding=1))
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        res = self.sa(x)
        # # 使用插值调整大小
        # res = F.interpolate(res, size=x.shape[2:], mode='bilinear', align_corners=False)
        res = self.dropout(self.sigmoid(res.expand_as(x)))
        high_precip_mask = torch.sigmoid((x > 0.15).float())  # 0.15为降水阈值
        return res * high_precip_mask  # 仅保留强降水区域的注意力


class TemporalAttention(nn.Module):
    def __init__(self, channel, reduction=16, num_layers=3):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        gate_channels = [channel]
        gate_channels += [channel // reduction] * num_layers
        gate_channels += [channel]

        self.ca = nn.Sequential()
        self.ca.add_module('flatten', Flatten())
        for i in range(len(gate_channels) - 2):
            self.ca.add_module('fc%d' % i, nn.Linear(gate_channels[i], gate_channels[i + 1]))
            self.ca.add_module('relu%d' % i, nn.PReLU())
        self.ca.add_module('last_fc', nn.Linear(gate_channels[-2], gate_channels[-1]))
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        res = self.avgpool(x)
        res = self.ca(res)
        res = res.unsqueeze(-1).unsqueeze(-1).expand_as(x)

        high_precip_mask = torch.sigmoid((x > 0.1).float())
        res = self.dropout(self.sigmoid(res))
        return res * high_precip_mask

class PixelAttention(nn.Module):
    def __init__(self, dim,reduction=16):
        super(PixelAttention, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0),
            nn.BatchNorm2d(dim // reduction),
            nn.PReLU(),
            nn.Conv2d(dim // reduction, dim, 1, padding=0),
        )
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        y = self.pa(x)
        y = self.sigmoid(y)
        return self.dropout(y)

class TSPFM(nn.Module):
    def __init__(self, input_channels,reduction_ratio=16,num_layers=3):
        super().__init__()
        self.temporal_att = TemporalAttention(input_channels,reduction=reduction_ratio,num_layers=num_layers)
        self.spatial_att = SpatialAttention(input_channels,reduction=reduction_ratio,num_layers=num_layers)
        self.pixel_att = PixelAttention(input_channels,reduction=reduction_ratio)

        self.fusion = nn.Sequential(
            nn.Conv2d(input_channels * 3, input_channels, 3,padding=1,groups=input_channels),
            nn.BatchNorm2d(input_channels),
            nn.PReLU(),
            nn.Dropout(0.2),
        )

        # self.conv = nn.Conv2d(input_channels, input_channels, 3, 1, 1,groups=input_channels)
        self.sigmoid = nn.Sigmoid()
        # self.drop = nn.Dropout(0.1)

    def forward(self, x):
        out_c = self.temporal_att(x)
        out_s = self.spatial_att(x)
        out_p = self.pixel_att(x)
        out_f = self.fusion(torch.cat([out_c,out_s,out_p], dim=1))
        # out_f = self.fusion(torch.cat([out_c,out_p], dim=1))

        # weight = self.sigmoid(out)
        # out = x * weight
        # out = self.conv(out_f)
        return self.sigmoid(out_f) * x
        
    def get_individual_attentions(self, x):
        """获取每个注意力模块的单独输出
        
        Args:
            x: 输入张量，形状为[B,C,H,W]，其中C表示时间步
        
        Returns:
            Dict包含三个注意力模块的输出：
            - temporal_attention: 时间注意力权重
            - spatial_attention: 空间注意力权重
            - pixel_attention: 像素注意力权重
        """
        temporal_att = self.temporal_att(x)
        spatial_att = self.spatial_att(x)
        pixel_att = self.pixel_att(x)
        
        return {
            'temporal_attention': temporal_att,
            'spatial_attention': spatial_att,
            'pixel_attention': pixel_att
        }
        
    def visualize_individual_attentions(self, x, device='cpu', debug=False):
        """可视化每个注意力模块的热图
        
        Args:
            x: 输入张量，形状为[B,C,H,W]，其中C表示时间步
            device: 运行设备
            debug: 是否输出调试信息
        
        Returns:
            Dict包含三个注意力模块的可视化热图
        """
        att_outputs = self.get_individual_attentions(x)
        visualizations = {}
        
        # 对每个时间步计算平均注意力权重
        for att_type, att_map in att_outputs.items():
            # [B,C,H,W] -> [B,H,W]，在时间维度上取平均
            avg_att_map = torch.mean(att_map, dim=1)
            
            # 如果是调试模式，打印注意力权重的统计信息
            if debug and att_type == 'temporal_attention':
                print(f"=== 时间注意力调试信息 ===")
                print(f"原始注意力权重范围: [{torch.min(att_map):.6f}, {torch.max(att_map):.6f}]")
                print(f"原始注意力权重均值: {torch.mean(att_map):.6f}")
                print(f"平均后注意力权重范围: [{torch.min(avg_att_map):.6f}, {torch.max(avg_att_map):.6f}]")
                print(f"平均后注意力权重均值: {torch.mean(avg_att_map):.6f}")
                
                # 计算强降水掩码的统计信息（模拟TemporalAttention中的计算）
                high_precip_mask = torch.sigmoid((x > 0.15).float()).mean(dim=(2,3), keepdim=True)
                print(f"强降水掩码均值: {torch.mean(high_precip_mask):.6f}")
                
                # 检查有多少注意力值大于0.1（任意阈值）
                non_zero_count = torch.sum(avg_att_map > 0.1).item()
                total_count = avg_att_map.numel()
                print(f"注意力值>0.1的比例: {non_zero_count/total_count*100:.2f}%")
        
            # 将注意力权重缩放到0-1范围
            min_val = torch.min(avg_att_map)
            max_val = torch.max(avg_att_map)
            normalized_att = (avg_att_map - min_val) / (max_val - min_val + 1e-8)
            visualizations[att_type] = normalized_att.cpu().detach().numpy()
        
        return visualizations
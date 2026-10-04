import math

import torch
import torch.nn as nn


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, ksize, stride, pad, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1):
        super().__init__()
        # Depthwise conv
        self.dconv = BaseConv(in_channels, in_channels, ksize=ksize, stride=stride, groups=in_channels)
        # Pointwise conv
        self.pconv = BaseConv(in_channels, out_channels, ksize=1, stride=1, groups=1)

    def forward(self, x):
        return self.pconv(self.dconv(x))

class Focus(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, 1)

    def forward(self, x):
        # x.shape: [B, C, H, W] -> [B, C*4, H/2, W/2]
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right), dim=1)
        return self.conv(x)

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, depthwise=False):
        super().__init__()
        hidden_channels = out_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, 1)
        
        # DWConv
        Conv = DWConv if depthwise else BaseConv
        self.conv2 = Conv(hidden_channels, out_channels, 3, 1)
        
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        return x + out if self.use_add else out

class CSPLayer(nn.Module):
    """Cross Stage Partial Block"""
    def __init__(self, in_channels, out_channels, num_bottlenecks=1, shortcut=True, depthwise=False):
        super().__init__()
        hidden_channels = out_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, 1)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, 1)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, 1)
        
        self.bottlenecks = nn.Sequential(*[
            Bottleneck(hidden_channels, hidden_channels, shortcut, depthwise) for _ in range(num_bottlenecks)
        ])

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.bottlenecks(x_1)
        out = torch.cat((x_1, x_2), dim=1)
        return self.conv3(out)

class SPPBottleneck(nn.Module):
    """Spatial Pyramid Pooling"""
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13)):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2) for k in kernel_sizes])
        self.conv2 = BaseConv(hidden_channels * (len(kernel_sizes) + 1), out_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        features = [x] + [m(x) for m in self.m]
        return self.conv2(torch.cat(features, dim=1))

class CSPDarknet(nn.Module):
    def __init__(self, depth, width, depthwise=False):
        super().__init__()
        base_channels = int(width * 64)
        base_depth = max(round(depth * 3), 1)
        self.stem = Focus(3, base_channels, ksize=3)
        
        Conv = DWConv if depthwise else BaseConv

        # Dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2),
            CSPLayer(base_channels * 2, base_channels * 2, num_bottlenecks=base_depth, depthwise=depthwise)
        )
        
        # Dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2),
            CSPLayer(base_channels * 4, base_channels * 4, num_bottlenecks=base_depth * 3, depthwise=depthwise)
        )
        
        # Dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2),
            CSPLayer(base_channels * 8, base_channels * 8, num_bottlenecks=base_depth * 3, depthwise=depthwise)
        )
        
        # Dark5 (SPP)
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2),
            SPPBottleneck(base_channels * 16, base_channels * 16),
            CSPLayer(base_channels * 16, base_channels * 16, num_bottlenecks=base_depth, shortcut=False, depthwise=depthwise)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.dark2(x)
        out_dark3 = self.dark3(x)  # Feature map 1
        out_dark4 = self.dark4(out_dark3)  # Feature map 2
        out_dark5 = self.dark5(out_dark4)  # Feature map 3
        return out_dark3, out_dark4, out_dark5

class YOLOPAFPN(nn.Module):
    def __init__(self, depth, width, depthwise=False):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise)
        in_channels = [256, 512, 1024]
        in_channels = [int(c * width) for c in in_channels]
        
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        Conv = DWConv if depthwise else BaseConv
        
        # Top-down FPN
        self.lateral_conv0 = BaseConv(in_channels[2], in_channels[1], 1, 1)
        self.C3_p4 = CSPLayer(in_channels[1] * 2, in_channels[1], max(round(3 * depth), 1), False, depthwise=depthwise)
        
        self.reduce_conv1 = BaseConv(in_channels[1], in_channels[0], 1, 1)
        self.C3_p3 = CSPLayer(in_channels[0] * 2, in_channels[0], max(round(3 * depth), 1), False, depthwise=depthwise)

        # Bottom-up PAN
        self.bu_conv2 = Conv(in_channels[0], in_channels[0], 3, 2)
        self.C3_n3 = CSPLayer(in_channels[0] * 2, in_channels[1], max(round(3 * depth), 1), False, depthwise=depthwise)

        self.bu_conv1 = Conv(in_channels[1], in_channels[1], 3, 2)
        self.C3_n4 = CSPLayer(in_channels[1] * 2, in_channels[2], max(round(3 * depth), 1), False, depthwise=depthwise)

    def forward(self, x):
        out_features = self.backbone(x)
        [x2, x1, x0] = out_features # x2: Dark3, x1: Dark4, x0: Dark5

        # Top-down
        fpn_out0 = self.lateral_conv0(x0)
        f_out0 = self.upsample(fpn_out0)
        f_out0 = torch.cat([f_out0, x1], 1)
        f_out0 = self.C3_p4(f_out0)

        fpn_out1 = self.reduce_conv1(f_out0)
        f_out1 = self.upsample(fpn_out1)
        f_out1 = torch.cat([f_out1, x2], 1)
        pan_out2 = self.C3_p3(f_out1) 

        # Bottom-up
        p_out1 = self.bu_conv2(pan_out2)
        p_out1 = torch.cat([p_out1, fpn_out1], 1)
        pan_out1 = self.C3_n3(p_out1) 

        p_out0 = self.bu_conv1(pan_out1)
        p_out0 = torch.cat([p_out0, fpn_out0], 1)
        pan_out0 = self.C3_n4(p_out0) 

        return pan_out2, pan_out1, pan_out0

class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width, depthwise=False):
        super().__init__()
        self.num_classes = num_classes
        in_channels = [int(c * width) for c in [256, 512, 1024]]
        
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(BaseConv(in_channels[i], int(256 * width), 1, 1))
            
            self.cls_convs.append(nn.Sequential(*[
                Conv(int(256 * width), int(256 * width), 3, 1),
                Conv(int(256 * width), int(256 * width), 3, 1),
            ]))
            self.reg_convs.append(nn.Sequential(*[
                Conv(int(256 * width), int(256 * width), 3, 1),
                Conv(int(256 * width), int(256 * width), 3, 1),
            ]))
            
            self.cls_preds.append(nn.Conv2d(int(256 * width), self.num_classes, 1, 1, 0))
            self.reg_preds.append(nn.Conv2d(int(256 * width), 4, 1, 1, 0))
            self.obj_preds.append(nn.Conv2d(int(256 * width), 1, 1, 1, 0))

    def forward(self, xin):
        outputs = []
        for k, x in enumerate(xin):
            x = self.stems[k](x)
            
            cls_feat = self.cls_convs[k](x)
            cls_output = self.cls_preds[k](cls_feat)
            
            reg_feat = self.reg_convs[k](x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)
            
            output = torch.cat([reg_output, obj_output, cls_output], dim=1)
            outputs.append(output)
            
        return outputs

    def initialize_biases(self, prior_probability=0.01):
        """Initialize foreground logits with a conservative YOLOX-style prior."""
        bias = -math.log((1.0 - prior_probability) / prior_probability)
        for prediction in list(self.obj_preds) + list(self.cls_preds):
            nn.init.constant_(prediction.bias, bias)

class ObjectDetectionModel(nn.Module):
    """
    - default: depth=0.33, width=0.50, depthwise=False
    """
    def __init__(self, num_classes, depth=0.33, width=0.50, depthwise=False, end2end=True):
        super().__init__()
        self.backbone = YOLOPAFPN(depth, width, depthwise=depthwise)
        self.head = YOLOXHead(num_classes, width, depthwise=depthwise)
        self.head.initialize_biases()
        self.end2end = bool(end2end)
        self.one2one_head = YOLOXHead(num_classes, width, depthwise=depthwise) if self.end2end else None
        if self.one2one_head is not None:
            # Start the auxiliary branch from the trained one-to-many head.
            for name in ("stems", "cls_convs", "reg_convs", "cls_preds", "reg_preds", "obj_preds"):
                for target, source in zip(getattr(self.one2one_head, name), getattr(self.head, name)):
                    target.load_state_dict(source.state_dict())

    def forward(self, x, return_one2one=False):
        fpn_outs = self.backbone(x)
        outputs = self.head(fpn_outs)
        if self.one2one_head is not None:
            one2one_outputs = self.one2one_head(fpn_outs)
            if return_one2one:
                return outputs, one2one_outputs
            return one2one_outputs
        return outputs

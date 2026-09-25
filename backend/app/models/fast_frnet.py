"""Fast-FRNet (Frustum-Range Networks for Scalable LiDAR Segmentation, IEEE TIP 2025).

Provides a pure PyTorch implementation of the Fast-FRNet architecture, supporting both:
1. RELLIS-3D Fine-Tuned Model (32 beams, H=32, W=512, fov_up=15.0 deg, fov_down=-25.0 deg)
2. SemanticKITTI Pretrained Model (64 beams, H=64, W=512, fov_up=3.0 deg, fov_down=-25.0 deg)

Maintains 100% parameter compatibility with official MMDeploy/MMDetection3D checkpoints
while operating with zero external C++ extension requirements.
"""

from typing import Tuple, List, Sequence, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Residual basic block for FRNet backbone stages."""

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.Hardswish(inplace=True)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ConvModule(nn.Module):
    """Convolution + Normalization + Activation building block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activate = nn.Hardswish(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activate(self.bn(self.conv(x)))


def make_res_layer(
    inplanes: int,
    planes: int,
    num_blocks: int,
    stride: int = 1,
    dilation: int = 1,
) -> nn.Sequential:
    """Build a sequential residual stage containing num_blocks BasicBlocks."""
    downsample = None
    if stride != 1 or inplanes != planes:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes),
        )

    layers = [
        BasicBlock(
            inplanes=inplanes,
            planes=planes,
            stride=stride,
            dilation=dilation,
            downsample=downsample,
        )
    ]
    for _ in range(1, num_blocks):
        layers.append(
            BasicBlock(
                inplanes=planes,
                planes=planes,
                stride=1,
                dilation=dilation,
            )
        )
    return nn.Sequential(*layers)


class FrustumFeatureEncoder(nn.Module):
    """Frustum Feature Encoder (FFE) extracting point and frustum-voxel features."""

    def __init__(
        self,
        in_channels: int = 4,
        feat_channels: Sequence[int] = (64, 128, 256, 256),
        feat_compression: int = 16,
    ) -> None:
        super().__init__()
        # 4 (x,y,z,i) + 1 (depth) + 3 (cluster center delta) = 8
        full_in = in_channels + 1 + 3
        self.pre_norm = nn.BatchNorm1d(full_in)

        all_channels = [full_in] + list(feat_channels)
        ffe = []
        for i in range(len(all_channels) - 1):
            cin, cout = all_channels[i], all_channels[i + 1]
            if i == len(all_channels) - 2:
                ffe.append(nn.Linear(cin, cout))
            else:
                ffe.append(
                    nn.Sequential(
                        nn.Linear(cin, cout, bias=False),
                        nn.BatchNorm1d(cout),
                        nn.ReLU(inplace=True),
                    )
                )
        self.ffe_layers = nn.ModuleList(ffe)
        self.compression_layers = nn.Sequential(
            nn.Linear(all_channels[-1], feat_compression),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        points: torch.Tensor,
        coors: torch.Tensor,
        precomputed_voxel_coors: Optional[torch.Tensor] = None,
        precomputed_inverse_map: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Encode raw LiDAR coordinates into point features and frustum voxel features.

        Args:
            points: (N, 4) tensor [x, y, z, intensity]
            coors: (N, 3) tensor [batch_idx, y_grid, x_grid]
            precomputed_voxel_coors: Optional precomputed unique voxel coordinates
            precomputed_inverse_map: Optional precomputed inverse mapping tensor

        Returns:
            voxel_feats: (num_unique_voxels, feat_compression)
            voxel_coors: (num_unique_voxels, 3)
            point_feats: List of intermediate point feature tensors
        """
        if precomputed_voxel_coors is not None and precomputed_inverse_map is not None:
            voxel_coors = precomputed_voxel_coors
            inverse_map = precomputed_inverse_map
        else:
            max_y = int(coors[:, 1].max().item()) + 1
            max_x = int(coors[:, 2].max().item()) + 1
            flat_key = coors[:, 0] * (max_y * max_x) + coors[:, 1] * max_x + coors[:, 2]
            u_keys, inverse_map = torch.unique(flat_key, return_inverse=True)
            voxel_coors = torch.stack(
                [u_keys // (max_y * max_x), (u_keys % (max_y * max_x)) // max_x, u_keys % max_x],
                dim=1,
            )
        num_voxels = voxel_coors.shape[0]

        # 1. Distance feature
        points_dist = torch.norm(points[:, :3], p=2, dim=1, keepdim=True)

        # 2. Vectorized cluster center feature decoration
        sum_feats = torch.zeros(
            (num_voxels, points.shape[1]), dtype=points.dtype, device=points.device
        )
        sum_feats.scatter_add_(
            0, inverse_map.unsqueeze(1).expand(-1, points.shape[1]), points
        )
        counts = torch.zeros(
            (num_voxels, 1), dtype=points.dtype, device=points.device
        )
        counts.scatter_add_(
            0,
            inverse_map.unsqueeze(1),
            torch.ones_like(inverse_map.unsqueeze(1), dtype=points.dtype),
        )
        voxel_mean = sum_feats / torch.clamp(counts, min=1.0)
        points_mean = voxel_mean[inverse_map]
        f_cluster = points[:, :3] - points_mean[:, :3]

        # 3. Concatenate and normalize features
        feats = torch.cat([points, points_dist, f_cluster], dim=-1)
        feats = self.pre_norm(feats)

        # 4. Multi-layer perceptron feature extraction
        point_feats = []
        for ffe in self.ffe_layers:
            feats = ffe(feats)
            point_feats.append(feats)

        # 5. Scatter-max pooling to voxel grid
        voxel_feats = torch.full(
            (num_voxels, feats.shape[1]),
            -float("inf"),
            dtype=feats.dtype,
            device=feats.device,
        )
        voxel_feats.scatter_reduce_(
            0,
            inverse_map.unsqueeze(1).expand(-1, feats.shape[1]),
            feats,
            reduce="amax",
            include_self=False,
        )

        # 6. Feature compression
        voxel_feats = self.compression_layers(voxel_feats)
        return voxel_feats, voxel_coors, point_feats


class FRNetBackbone(nn.Module):
    """FRNet 2D/3D Multi-Scale Fusion Backbone."""

    def __init__(
        self,
        in_channels: int = 16,
        point_in_channels: int = 384,
        output_shape: Sequence[int] = (32, 512),
        stem_channels: int = 128,
        num_stages: int = 4,
        out_channels: Sequence[int] = (128, 128, 128, 128),
        strides: Sequence[int] = (1, 2, 2, 2),
        dilations: Sequence[int] = (1, 1, 1, 1),
        fuse_channels: Sequence[int] = (256, 128),
    ) -> None:
        super().__init__()
        self.output_shape = output_shape
        self.ny, self.nx = output_shape[0], output_shape[1]

        # Stems
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels // 2),
            nn.Hardswish(inplace=True),
            nn.Conv2d(stem_channels // 2, stem_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.Hardswish(inplace=True),
            nn.Conv2d(stem_channels, stem_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.Hardswish(inplace=True),
        )
        self.point_stem = nn.Sequential(
            nn.Linear(point_in_channels, stem_channels, bias=False),
            nn.BatchNorm1d(stem_channels),
            nn.ReLU(inplace=True),
        )
        self.fusion_stem = nn.Sequential(
            nn.Conv2d(stem_channels * 2, stem_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.Hardswish(inplace=True),
        )

        # ResNet-34 stages
        stage_blocks = (3, 4, 6, 3)
        self.res_layers = []
        self.point_fusion_layers = nn.ModuleList()
        self.pixel_fusion_layers = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.strides = []

        inplanes = stem_channels
        overall_stride = 1
        for i, num_blocks in enumerate(stage_blocks):
            stride = strides[i]
            overall_stride = stride * overall_stride
            self.strides.append(overall_stride)
            planes = out_channels[i]

            res_layer = make_res_layer(
                inplanes=inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilations[i],
            )
            self.point_fusion_layers.append(
                nn.Sequential(
                    nn.Linear(inplanes + planes, planes, bias=False),
                    nn.BatchNorm1d(planes),
                    nn.ReLU(inplace=True),
                )
            )
            self.pixel_fusion_layers.append(
                nn.Sequential(
                    nn.Conv2d(planes * 2, planes, 3, padding=1, bias=False),
                    nn.BatchNorm2d(planes),
                    nn.Hardswish(inplace=True),
                )
            )
            self.attention_layers.append(
                nn.Sequential(
                    nn.Conv2d(planes, planes, 3, padding=1, bias=False),
                    nn.BatchNorm2d(planes),
                    nn.Hardswish(inplace=True),
                    nn.Conv2d(planes, planes, 3, padding=1, bias=False),
                    nn.BatchNorm2d(planes),
                    nn.Sigmoid(),
                )
            )

            inplanes = planes
            layer_name = f"layer{i + 1}"
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        # Final multi-scale fusion layers
        in_ch = stem_channels + sum(out_channels)
        self.fuse_layer1 = ConvModule(in_ch, fuse_channels[0])
        self.point_fuse_layer1 = nn.Sequential(
            nn.Linear(in_ch, fuse_channels[0], bias=False),
            nn.BatchNorm1d(fuse_channels[0]),
            nn.ReLU(inplace=True),
        )
        self.fuse_layer2 = ConvModule(fuse_channels[0], fuse_channels[1])
        self.point_fuse_layer2 = nn.Sequential(
            nn.Linear(fuse_channels[0], fuse_channels[1], bias=False),
            nn.BatchNorm1d(fuse_channels[1]),
            nn.ReLU(inplace=True),
        )

    def frustum2pixel(
        self,
        frustum_features: torch.Tensor,
        coors: torch.Tensor,
        batch_size: int = 1,
        stride: int = 1,
    ) -> torch.Tensor:
        nx = self.nx // stride
        ny = self.ny // stride
        pixel_features = torch.zeros(
            (batch_size, ny, nx, frustum_features.shape[-1]),
            dtype=frustum_features.dtype,
            device=frustum_features.device,
        )
        pixel_features[coors[:, 0], coors[:, 1], coors[:, 2]] = frustum_features
        return pixel_features.permute(0, 3, 1, 2).contiguous()

    def pixel2point(
        self,
        pixel_features: torch.Tensor,
        coors: torch.Tensor,
        stride: int = 1,
    ) -> torch.Tensor:
        pixel_features = pixel_features.permute(0, 2, 3, 1).contiguous()
        return pixel_features[
            coors[:, 0], coors[:, 1] // stride, coors[:, 2] // stride
        ]

    def point2frustum(
        self,
        point_features: torch.Tensor,
        pts_coors: torch.Tensor,
        stride: int = 1,
        precomputed_map: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if precomputed_map is not None:
            voxel_coors, inverse_map = precomputed_map
        else:
            nx = self.nx // stride
            ny = self.ny // stride
            flat_key = (
                pts_coors[:, 0] * (ny * nx)
                + (pts_coors[:, 1] // stride) * nx
                + (pts_coors[:, 2] // stride)
            )
            u_keys, inverse_map = torch.unique(flat_key, return_inverse=True)
            voxel_coors = torch.stack(
                [u_keys // (ny * nx), (u_keys % (ny * nx)) // nx, u_keys % nx],
                dim=1,
            )
        num_v = voxel_coors.shape[0]

        frustum_features = torch.full(
            (num_v, point_features.shape[1]),
            -float("inf"),
            dtype=point_features.dtype,
            device=point_features.device,
        )
        frustum_features.scatter_reduce_(
            0,
            inverse_map.unsqueeze(1).expand(-1, point_features.shape[1]),
            point_features,
            reduce="amax",
            include_self=False,
        )
        return voxel_coors, frustum_features

    def forward(
        self,
        voxel_feats: torch.Tensor,
        voxel_coors: torch.Tensor,
        point_feats_list: List[torch.Tensor],
        pts_coors: torch.Tensor,
        stride_maps: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        point_feats = point_feats_list[-1]
        batch_size = pts_coors[-1, 0].item() + 1

        # Precompute/reuse stride maps for all strides in a single fast vectorized pass
        if stride_maps is None:
            stride_maps = {}
            for s in set([1] + list(self.strides)):
                nx = self.nx // s
                ny = self.ny // s
                flat_key = (
                    pts_coors[:, 0] * (ny * nx)
                    + (pts_coors[:, 1] // s) * nx
                    + (pts_coors[:, 2] // s)
                )
                u_keys, inverse_map = torch.unique(flat_key, return_inverse=True)
                vc = torch.stack(
                    [u_keys // (ny * nx), (u_keys % (ny * nx)) // nx, u_keys % nx],
                    dim=1,
                )
                stride_maps[s] = (vc, inverse_map)

        x = self.frustum2pixel(voxel_feats, voxel_coors, batch_size, stride=1)
        x = self.stem(x)
        map_point_feats = self.pixel2point(x, pts_coors, stride=1)
        fusion_point_feats = torch.cat((map_point_feats, point_feats), dim=1)
        point_feats = self.point_stem(fusion_point_feats)

        stride_voxel_coors, frustum_feats = self.point2frustum(
            point_feats, pts_coors, stride=1, precomputed_map=stride_maps[1]
        )
        pixel_feats = self.frustum2pixel(
            frustum_feats, stride_voxel_coors, batch_size, stride=1
        )
        fusion_pixel_feats = torch.cat((pixel_feats, x), dim=1)
        x = self.fusion_stem(fusion_pixel_feats)

        outs = [x]
        out_points = [point_feats]
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)

            s = self.strides[i]
            # Frustum-to-point fusion
            map_point_feats = self.pixel2point(x, pts_coors, stride=s)
            fusion_point_feats = torch.cat((map_point_feats, point_feats), dim=1)
            point_feats = self.point_fusion_layers[i](fusion_point_feats)

            # Point-to-frustum fusion
            stride_voxel_coors, frustum_feats = self.point2frustum(
                point_feats, pts_coors, stride=s, precomputed_map=stride_maps[s]
            )
            pixel_feats = self.frustum2pixel(
                frustum_feats, stride_voxel_coors, batch_size, stride=s
            )
            fusion_pixel_feats = torch.cat((pixel_feats, x), dim=1)
            fuse_out = self.pixel_fusion_layers[i](fusion_pixel_feats)

            # Residual-attentive modulation
            attention_map = self.attention_layers[i](fuse_out)
            x = fuse_out * attention_map + x
            outs.append(x)
            out_points.append(point_feats)

        # Multi-scale feature alignment
        for i in range(len(outs)):
            if outs[i].shape != outs[0].shape:
                outs[i] = F.interpolate(
                    outs[i],
                    size=outs[0].size()[2:],
                    mode="bilinear",
                    align_corners=True,
                )

        outs[0] = torch.cat(outs, dim=1)
        out_points[0] = torch.cat(out_points, dim=1)

        outs[0] = self.fuse_layer1(outs[0])
        out_points[0] = self.point_fuse_layer1(out_points[0])
        outs[0] = self.fuse_layer2(outs[0])
        out_points[0] = self.point_fuse_layer2(out_points[0])

        return outs[0], out_points[0]


class FRHead(nn.Module):
    """FRHead decoding point and multi-scale frustum features into semantic logits."""

    def __init__(
        self,
        in_channels: int = 128,
        middle_channels: Sequence[int] = (128, 256, 128, 64),
        num_classes: int = 20,
    ) -> None:
        super().__init__()
        mlps = []
        cur_in = in_channels
        for out_ch in middle_channels:
            mlps.append(
                nn.Sequential(
                    nn.Linear(cur_in, out_ch, bias=False),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )
            cur_in = out_ch
        self.mlps = nn.ModuleList(mlps)
        self.conv_seg = nn.Linear(cur_in, num_classes)

    def forward(
        self,
        voxel_feats: torch.Tensor,
        point_feats_backbone: torch.Tensor,
        point_feats_list: List[torch.Tensor],
        pts_coors: torch.Tensor,
    ) -> torch.Tensor:
        voxel_feats = voxel_feats.permute(0, 2, 3, 1)
        map_point_feats = voxel_feats[
            pts_coors[:, 0], pts_coors[:, 1], pts_coors[:, 2]
        ]
        point_feats = point_feats_list[:-1]

        for i, mlp in enumerate(self.mlps):
            map_point_feats = mlp(map_point_feats)
            if i == 0:
                map_point_feats = map_point_feats + point_feats_backbone
            else:
                map_point_feats = map_point_feats + point_feats[-i]

        return self.conv_seg(map_point_feats)


class FastFRNet(nn.Module):
    """Complete Fast-FRNet Neural Segmentation Model."""

    def __init__(
        self,
        output_shape: Sequence[int] = (32, 512),
        fov_up: float = 15.0,
        fov_down: float = -25.0,
        num_classes: int = 20,
    ) -> None:
        super().__init__()
        self.output_shape = output_shape
        self.H, self.W = output_shape[0], output_shape[1]
        self.fov_up = fov_up
        self.fov_down = fov_down
        self.num_classes = num_classes

        self.voxel_encoder = FrustumFeatureEncoder()
        self.backbone = FRNetBackbone(output_shape=output_shape)
        self.decode_head = FRHead(num_classes=num_classes)

    def preprocess(self, points: torch.Tensor) -> torch.Tensor:
        """Compute spherical range coordinate indices (H, W) for each point."""
        depth = torch.linalg.norm(points[:, :3], ord=2, dim=1)
        yaw = -torch.atan2(points[:, 1], points[:, 0])
        pitch = torch.arcsin(
            torch.clamp(points[:, 2] / torch.clamp(depth, min=1e-4), -1.0, 1.0)
        )

        fov_up_rad = self.fov_up / 180.0 * np.pi
        fov_down_rad = self.fov_down / 180.0 * np.pi
        fov_rad = abs(fov_down_rad) + abs(fov_up_rad)

        coors_x = 0.5 * (yaw / np.pi + 1.0) * self.W
        coors_y = (1.0 - (pitch + abs(fov_down_rad)) / fov_rad) * self.H

        coors_x = torch.clamp(
            torch.floor(coors_x).to(torch.int64), min=0, max=self.W - 1
        )
        coors_y = torch.clamp(
            torch.floor(coors_y).to(torch.int64), min=0, max=self.H - 1
        )
        coors = torch.stack([torch.zeros_like(coors_y), coors_y, coors_x], dim=1)
        return coors

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """Forward pass executing spherical projection and neural segmentation.

        Args:
            points: (N, 4) tensor containing float32 [x, y, z, intensity]

        Returns:
            logits: (N, num_classes) tensor
        """
        coors = self.preprocess(points)

        # Precompute 1D flat key unique maps for all strides (1, 2, 4, 8) once
        stride_maps = {}
        for s in [1, 2, 4, 8]:
            cur_H = self.H // s
            cur_W = self.W // s
            flat_key = (
                coors[:, 0] * (cur_H * cur_W)
                + (coors[:, 1] // s) * cur_W
                + (coors[:, 2] // s)
            )
            u_keys, inverse_map = torch.unique(flat_key, return_inverse=True)
            voxel_coors = torch.stack(
                [
                    u_keys // (cur_H * cur_W),
                    (u_keys % (cur_H * cur_W)) // cur_W,
                    u_keys % cur_W,
                ],
                dim=1,
            )
            stride_maps[s] = (voxel_coors, inverse_map)

        s1_coors, s1_inv = stride_maps[1]
        voxel_feats, voxel_coors, point_feats_list = self.voxel_encoder(
            points,
            coors,
            precomputed_voxel_coors=s1_coors,
            precomputed_inverse_map=s1_inv,
        )
        voxel_out, point_out = self.backbone(
            voxel_feats, voxel_coors, point_feats_list, coors, stride_maps=stride_maps
        )
        logits = self.decode_head(voxel_out, point_out, point_feats_list, coors)
        return logits

    def predict(self, points: torch.Tensor) -> torch.Tensor:
        """Predict per-point semantic class indices."""
        with torch.no_grad():
            logits = self.forward(points)
            return torch.argmax(logits, dim=1)

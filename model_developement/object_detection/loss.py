import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou

class ObjectDetectionLoss(nn.Module):
    def __init__(self, num_classes, strides=[8, 16, 32], obj_focal_gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.giou_loss = GIoUloss(reduction="none")
        self.obj_focal_gamma = float(obj_focal_gamma)

    def objectness_loss(self, logits, targets):
        """BCE with focal modulation, emphasizing hard background mistakes."""
        loss = self.bcewithlog_loss(logits, targets)
        if self.obj_focal_gamma > 0:
            probabilities = logits.sigmoid()
            p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
            loss = loss * (1.0 - p_t).pow(self.obj_focal_gamma)
        return loss

    def one2one_loss(self, preds, targets):
        """End-to-end set loss with Hungarian one-prediction-per-GT matching."""
        from scipy.optimize import linear_sum_assignment

        outputs, _, _, _ = self.decode_outputs(preds)
        bbox_preds = outputs[..., :4].float()
        obj_logits = outputs[..., 4].float()
        cls_logits = outputs[..., 5:].float()
        batch_size, num_predictions = obj_logits.shape
        total_box = outputs.new_zeros((), dtype=torch.float32)
        total_cls = outputs.new_zeros((), dtype=torch.float32)
        total_obj = outputs.new_zeros((), dtype=torch.float32)
        total_matches = 0

        for batch_idx in range(batch_size):
            target = targets[batch_idx]
            obj_target = torch.zeros_like(obj_logits[batch_idx])
            if len(target):
                image_h = preds[0].shape[2] * self.strides[0]
                image_w = preds[0].shape[3] * self.strides[0]
                scale = target.new_tensor([image_w, image_h, image_w, image_h])
                gt_boxes = target[:, 1:5].float() * scale
                gt_classes = target[:, 0].long()
                pred_boxes = bbox_preds[batch_idx]

                # Match only feature points inside each GT or its nearby center.
                strides = torch.cat([
                    torch.full((level.shape[2] * level.shape[3],), stride,
                               device=outputs.device, dtype=torch.float32)
                    for level, stride in zip(preds, self.strides)
                ])
                centers = []
                for level, stride in zip(preds, self.strides):
                    h, w = level.shape[2:]
                    yy, xx = torch.meshgrid(torch.arange(h, device=outputs.device),
                                            torch.arange(w, device=outputs.device), indexing="ij")
                    centers.append(torch.stack(((xx.reshape(-1) + 0.5) * stride,
                                                (yy.reshape(-1) + 0.5) * stride), dim=1))
                centers = torch.cat(centers).float()
                delta = (centers[None] - gt_boxes[:, None, :2]).abs()
                half = gt_boxes[:, None, 2:] / 2
                in_box = (delta <= half).all(-1)
                radius = 2.5 * strides[None, :]
                in_center = (delta <= radius[..., None]).all(-1)
                candidate_mask = (in_box | in_center).any(0)
                if not candidate_mask.any():
                    candidate_mask[((centers - gt_boxes[:, None, :2]).square().sum(-1)).argmin(1)] = True
                candidate_ids = candidate_mask.nonzero(as_tuple=False).flatten()
                candidate_boxes = pred_boxes[candidate_ids]

                pred_xyxy = torch.cat((candidate_boxes[:, :2] - candidate_boxes[:, 2:] / 2,
                                       candidate_boxes[:, :2] + candidate_boxes[:, 2:] / 2), dim=1)
                gt_xyxy = torch.cat((gt_boxes[:, :2] - gt_boxes[:, 2:] / 2,
                                     gt_boxes[:, :2] + gt_boxes[:, 2:] / 2), dim=1)
                with torch.no_grad():
                    class_prob = (
                        cls_logits[batch_idx, candidate_ids].sigmoid()
                        * obj_logits[batch_idx, candidate_ids].sigmoid().unsqueeze(1)
                    )
                    class_cost = -class_prob[:, gt_classes]
                    iou_cost = -generalized_box_iou(pred_xyxy, gt_xyxy)
                    cost = class_cost + 3.0 * iou_cost
                    row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())
                    pred_ind = candidate_ids[torch.as_tensor(row_ind, device=outputs.device, dtype=torch.long)]
                    gt_ind = torch.as_tensor(col_ind, device=outputs.device, dtype=torch.long)

                obj_target[pred_ind] = 1.0
                matched_boxes = bbox_preds[batch_idx, pred_ind]
                matched_gt = gt_boxes[gt_ind]
                total_box += self.giou_loss(matched_boxes, matched_gt).sum()
                total_cls += self.bcewithlog_loss(
                    cls_logits[batch_idx, pred_ind],
                    F.one_hot(gt_classes[gt_ind], self.num_classes).float(),
                ).sum()
                total_matches += len(pred_ind)

            total_obj += self.objectness_loss(obj_logits[batch_idx], obj_target).mean()

        normalizer = max(total_matches, 1)
        box_loss = total_box / normalizer * 5.0
        cls_loss = total_cls / normalizer
        obj_loss = total_obj / max(batch_size, 1)
        return box_loss + cls_loss + obj_loss, {
            "loss_giou": box_loss.detach(),
            "loss_obj": obj_loss.detach(),
            "loss_cls": cls_loss.detach(),
            "num_matches": torch.tensor(total_matches, device=outputs.device),
        }

    def forward(self, preds, targets):
        outputs, x_shifts, y_shifts, expanded_strides = self.decode_outputs(preds)
        
        bbox_preds = outputs[:, :, :4]  
        obj_preds = outputs[:, :, 4].unsqueeze(-1) 
        cls_preds = outputs[:, :, 5:]   

        batch_size = outputs.shape[0]
        loss_cls = torch.zeros(1, device=outputs.device)
        loss_giou = torch.zeros(1, device=outputs.device)
        loss_obj = torch.zeros(1, device=outputs.device)
        num_fg = torch.zeros((), dtype=torch.long, device=outputs.device)

        for batch_idx in range(batch_size):
            num_gt = len(targets[batch_idx])
            if num_gt == 0: 
                loss_obj += self.objectness_loss(
                    obj_preds[batch_idx], torch.zeros_like(obj_preds[batch_idx])
                ).sum()
                continue
                
            img_size = [preds[0].shape[2] * self.strides[0], preds[0].shape[3] * self.strides[0]]
            gt_bboxes_per_image = targets[batch_idx][:, 1:5] * torch.tensor([img_size[1], img_size[0], img_size[1], img_size[0]], device=outputs.device)
            gt_classes = targets[batch_idx][:, 0]

            bboxes_preds_per_image = bbox_preds[batch_idx]
            cls_preds_per_image = cls_preds[batch_idx]
            obj_preds_per_image = obj_preds[batch_idx]

            with torch.no_grad():
                gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img = self.get_assignments(
                    batch_idx, num_gt, gt_bboxes_per_image, gt_classes,
                    bboxes_preds_per_image, cls_preds_per_image, obj_preds_per_image,
                    expanded_strides, x_shifts, y_shifts
                )
            
            num_fg += num_fg_img
            obj_target = fg_mask.unsqueeze(-1)

            cls_target = F.one_hot(
                gt_matched_classes.to(torch.int64), self.num_classes
            ) * pred_ious_this_matching.unsqueeze(-1)
            loss_giou += self.giou_loss(
                bboxes_preds_per_image[fg_mask],
                gt_bboxes_per_image[matched_gt_inds],
            ).sum()
            loss_cls += self.bcewithlog_loss(
                cls_preds_per_image[fg_mask], cls_target
            ).sum()

            loss_obj += self.objectness_loss(
                obj_preds_per_image, obj_target.type(torch.float32)
            ).sum()

        num_fg = num_fg.clamp(min=1).to(dtype=loss_giou.dtype)
        loss_giou = (loss_giou / num_fg) * 5.0
        loss_obj = (loss_obj / num_fg) * 1.0
        loss_cls = (loss_cls / num_fg) * 1.0
        
        total_loss = loss_giou + loss_obj + loss_cls
        # Detached tensors avoid synchronizing CUDA with the host every batch.
        return total_loss, {
            "loss_giou": loss_giou.detach(),
            "loss_obj": loss_obj.detach(),
            "loss_cls": loss_cls.detach(),
        }

    def decode_outputs(self, preds):
        outputs = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, stride in enumerate(self.strides):
            pred = preds[k] 
            B, C, H, W = pred.shape
            
            yv, xv = torch.meshgrid([torch.arange(H), torch.arange(W)], indexing='ij')
            grid = torch.stack((xv, yv), 2).view(1, H * W, 2).type_as(pred)
            
            pred = pred.view(B, C, H * W).permute(0, 2, 1)
            xy = (pred[..., :2] + grid) * stride
            wh = torch.exp(pred[..., 2:4]) * stride
            pred = torch.cat([xy, wh, pred[..., 4:]], dim=-1)

            outputs.append(pred)
            x_shifts.append(grid[..., 0])
            y_shifts.append(grid[..., 1])
            expanded_strides.append(torch.zeros(1, grid.shape[1]).fill_(stride).type_as(pred))

        return torch.cat(outputs, 1), torch.cat(x_shifts, 1), torch.cat(y_shifts, 1), torch.cat(expanded_strides, 1)

    @torch.no_grad()
    def get_assignments(self, batch_idx, num_gt, gt_bboxes_per_image, gt_classes, bboxes_preds_per_image, cls_preds_per_image, obj_preds_per_image, expanded_strides, x_shifts, y_shifts):
        num_bboxes = bboxes_preds_per_image.shape[0]
        
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, num_bboxes, num_gt)

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds_per_image[fg_mask]
        obj_preds_ = obj_preds_per_image[fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if num_in_boxes_anchor == 0:
            device = bboxes_preds_per_image.device
            return (
                torch.zeros(0, dtype=gt_classes.dtype, device=device),
                torch.zeros(num_bboxes, dtype=torch.bool, device=device),
                torch.zeros(0, dtype=bboxes_preds_per_image.dtype, device=device),
                torch.zeros(0, dtype=torch.long, device=device),
                torch.zeros((), dtype=torch.long, device=device),
            )

        pair_wise_ious = self.bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image)
        
        gt_cls_per_image = F.one_hot(gt_classes.to(torch.int64), self.num_classes).float().unsqueeze(1).repeat(1, num_in_boxes_anchor, 1)
        cls_preds_ = cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_() * obj_preds_.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
        
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)
        pair_wise_cls_loss = F.binary_cross_entropy(cls_preds_.sqrt_(), gt_cls_per_image, reduction="none").sum(-1)
        
        cost = pair_wise_cls_loss + 3.0 * pair_wise_ious_loss + 100000.0 * (~is_in_boxes_and_center)

        num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds, fg_mask_inboxes = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt)
        
        fg_mask_copy = fg_mask.clone()
        true_indices = torch.where(fg_mask)[0]
        fg_mask_copy[true_indices] = fg_mask_inboxes

        return gt_matched_classes, fg_mask_copy, pred_ious_this_matching, matched_gt_inds, num_fg

    def get_in_boxes_info(self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts, total_num_anchors, num_gt):
        x_centers = (x_shifts + 0.5) * expanded_strides
        y_centers = (y_shifts + 0.5) * expanded_strides

        gt_x_center = gt_bboxes_per_image[:, 0].unsqueeze(1)
        gt_y_center = gt_bboxes_per_image[:, 1].unsqueeze(1)
        gt_w = gt_bboxes_per_image[:, 2].unsqueeze(1)
        gt_h = gt_bboxes_per_image[:, 3].unsqueeze(1)

        b_l = x_centers - (gt_x_center - gt_w / 2)
        b_r = (gt_x_center + gt_w / 2) - x_centers
        b_t = y_centers - (gt_y_center - gt_h / 2)
        b_b = (gt_y_center + gt_h / 2) - y_centers

        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], dim=2)
        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0 
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0       

        center_radius = 2.5
        c_l = x_centers - (gt_x_center - center_radius * expanded_strides)
        c_r = (gt_x_center + center_radius * expanded_strides) - x_centers
        c_t = y_centers - (gt_y_center - center_radius * expanded_strides)
        c_b = (gt_y_center + center_radius * expanded_strides) - y_centers

        center_deltas = torch.stack([c_l, c_t, c_r, c_b], dim=2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0 
        is_in_centers_all = is_in_centers.sum(dim=0) > 0       

        fg_mask = is_in_boxes_all | is_in_centers_all 

        is_in_boxes_and_center = is_in_boxes[:, fg_mask] & is_in_centers[:, fg_mask]

        return fg_mask, is_in_boxes_and_center
        
    def bboxes_iou(self, bboxes_a, bboxes_b):
        b1_x1 = bboxes_a[:, 0] - bboxes_a[:, 2] / 2
        b1_y1 = bboxes_a[:, 1] - bboxes_a[:, 3] / 2
        b1_x2 = bboxes_a[:, 0] + bboxes_a[:, 2] / 2
        b1_y2 = bboxes_a[:, 1] + bboxes_a[:, 3] / 2
        
        b2_x1 = bboxes_b[:, 0] - bboxes_b[:, 2] / 2
        b2_y1 = bboxes_b[:, 1] - bboxes_b[:, 3] / 2
        b2_x2 = bboxes_b[:, 0] + bboxes_b[:, 2] / 2
        b2_y2 = bboxes_b[:, 1] + bboxes_b[:, 3] / 2

        inter_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1.unsqueeze(0))
        inter_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1.unsqueeze(0))
        inter_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2.unsqueeze(0))
        inter_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2.unsqueeze(0))

        inter_w = torch.clamp(inter_x2 - inter_x1, min=0.0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0.0)
        inter_area = inter_w * inter_h

        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        
        union_area = b1_area.unsqueeze(1) + b2_area.unsqueeze(0) - inter_area + 1e-16

        return inter_area / union_area
        
    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt):
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_ks[gt_idx], largest=False)
            matching_matrix[gt_idx][pos_idx] = 1
            
        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
            
        fg_mask_inboxes = matching_matrix.sum(0) > 0
        
        num_fg = fg_mask_inboxes.sum()
        
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0) 
        
        gt_matched_classes = gt_classes[matched_gt_inds]
        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]
        
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds, fg_mask_inboxes

class GIoUloss(nn.Module):
    def __init__(self, reduction="none"):
        super(GIoUloss, self).__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        # Box geometry and epsilon handling are more stable in float32 under AMP.
        pred = pred.float()
        target = target.float()
        pred_x1 = pred[:, 0] - pred[:, 2] / 2
        pred_y1 = pred[:, 1] - pred[:, 3] / 2
        pred_x2 = pred[:, 0] + pred[:, 2] / 2
        pred_y2 = pred[:, 1] + pred[:, 3] / 2

        target_x1 = target[:, 0] - target[:, 2] / 2
        target_y1 = target[:, 1] - target[:, 3] / 2
        target_x2 = target[:, 0] + target[:, 2] / 2
        target_y2 = target[:, 1] + target[:, 3] / 2

        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)

        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_w * inter_h

        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        union_area = pred_area + target_area - inter_area + 1e-16

        eps = 1e-7
        iou = inter_area / union_area.clamp(min=eps)

        # GIoU provides a regression signal even when predicted and target
        # boxes do not overlap. Assignment and classification still use IoU.
        enclosing_x1 = torch.minimum(pred_x1, target_x1)
        enclosing_y1 = torch.minimum(pred_y1, target_y1)
        enclosing_x2 = torch.maximum(pred_x2, target_x2)
        enclosing_y2 = torch.maximum(pred_y2, target_y2)
        enclosing_area = (enclosing_x2 - enclosing_x1) * (enclosing_y2 - enclosing_y1)
        giou = iou - (enclosing_area - union_area) / enclosing_area.clamp(min=eps)
        loss = 1.0 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
            
        return loss

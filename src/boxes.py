import torch
import torchvision

def get_valid_boxes(current_boxes, past_boxes):
    valid_boxes = []
    for current_box in current_boxes:
        matched_frames = 0
        for past_frame_boxes in past_boxes[max(-10, -len(past_boxes)):]: 
            ious = torchvision.ops.box_iou(current_box.unsqueeze(0), past_frame_boxes)
            abs_diff = torch.abs(current_box - past_frame_boxes)
            if (ious > 0.3).any() or (abs_diff < 40).all():
                matched_frames += 1
                
        if matched_frames >= 1:
            valid_boxes.append(current_box)
    valid_boxes = torch.stack(valid_boxes) if valid_boxes else torch.empty((0, 4), device=current_boxes.device)
    return valid_boxes
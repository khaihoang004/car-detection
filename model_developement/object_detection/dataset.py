import os
import cv2
import json
import torch
import numpy as np
from torch.utils.data import Dataset

class ObjectDetectionDataset(Dataset):
    def __init__(self, data_dir, image_size=(360, 640), transform=None, image_files=None):
        self.data_dir = data_dir
        self.image_dir = os.path.join(data_dir, 'images')
        self.label_dir = os.path.join(data_dir, 'labels')
        
        h, w = image_size
        new_h = int(round(h / 32) * 32)
        new_w = int(round(w / 32) * 32)
        self.image_size = (new_h, new_w)
        
        if self.image_size != image_size:
            print(f"Warning: image_size {image_size} was adjusted to {self.image_size} to be a multiple of 32.")
        # --------------------------------------------------------

        self.transform = transform
        
        if not os.path.exists(self.image_dir):
            self.image_dir = data_dir
        if not os.path.exists(self.label_dir):
            self.label_dir = data_dir
            
        available_images = sorted(
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if image_files is None:
            self.image_files = available_images
        else:
            requested = set(image_files)
            missing = requested.difference(available_images)
            if missing:
                raise FileNotFoundError(f"Split references missing images, e.g. {sorted(missing)[:3]}")
            self.image_files = [name for name in available_images if name in requested]
        
        self.metadata = {}
        self._load_metadata()

    def _load_metadata(self):
        metadata_path = os.path.join(self.data_dir, "metadata.json")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Missing dataset metadata: {metadata_path}"
            )

        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

        classes = metadata.get("classes")
        if not classes:
            raise ValueError(
                "metadata.json must contain a non-empty 'classes' array."
            )

        self.metadata = metadata
        self.classes = metadata["classes"]
        self.class_map = {
            int(c["id"]): c["name"]
            for c in self.classes
        }

    def get_class_weights(self):
        class_counts = torch.zeros(self.num_classes)
        
        for img_name in self.image_files:
            label_name = os.path.splitext(img_name)[0] + '.txt'
            label_path = os.path.join(self.label_dir, label_name)
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            if class_id in self.class_map:
                                class_counts[class_id] += 1
                                
        total_objects = class_counts.sum()
        if total_objects == 0:
            return torch.ones(self.num_classes)
            
        safe_counts = torch.where(class_counts == 0, torch.tensor(1.0), class_counts)
        class_weights = total_objects / (self.num_classes * safe_counts)
        
        return class_weights

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)
        
        bboxes = []
        class_labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        if class_id not in self.class_map:
                            raise ValueError(
                                f"Unknown class id {class_id} in {label_path}"
                            )                        
                        x_c, y_c, bw, bh = map(float, parts[1:])

                        x_min = x_c - bw / 2.0
                        y_min = y_c - bh / 2.0
                        x_max = x_c + bw / 2.0
                        y_max = y_c + bh / 2.0

                        x_min = np.clip(x_min, 0.0, 1.0)
                        y_min = np.clip(y_min, 0.0, 1.0)
                        x_max = np.clip(x_max, 0.0, 1.0)
                        y_max = np.clip(y_max, 0.0, 1.0)

                        new_bw = x_max - x_min
                        new_bh = y_max - y_min
                        new_xc = x_min + new_bw / 2.0
                        new_yc = y_min + new_bh / 2.0

                        if new_bw > 0 and new_bh > 0:
                            bboxes.append([new_xc, new_yc, new_bw, new_bh])
                            class_labels.append(class_id)
                                
        if self.transform is not None:
            if len(bboxes) > 0:
                transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
                image_tensor = transformed['image']
                
                t_bboxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
                t_labels = torch.tensor(transformed['class_labels'], dtype=torch.float32).unsqueeze(1)
                
                if len(t_bboxes) > 0:
                    boxes_tensor = torch.cat((t_labels, t_bboxes), dim=1)
                else:
                    boxes_tensor = torch.empty((0, 5), dtype=torch.float32)
            else:
                transformed = self.transform(image=image, bboxes=[], class_labels=[])
                image_tensor = transformed['image']
                boxes_tensor = torch.empty((0, 5), dtype=torch.float32)
                
        else:
            image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
            image_tensor = torch.tensor(image / 255.0, dtype=torch.float32).permute(2, 0, 1)
            
            if len(bboxes) > 0:
                boxes_tensor = torch.tensor([
                    [c] + box for c, box in zip(class_labels, bboxes)
                ], dtype=torch.float32)
            else:
                boxes_tensor = torch.empty((0, 5), dtype=torch.float32)
            
        return image_tensor, boxes_tensor

    @property
    def num_classes(self):
        return len(self.classes)


def detection_collate_fn(batch):
    images = []
    targets = []
    
    for img, boxes in batch:
        images.append(img)
        targets.append(boxes)
        
    images = torch.stack(images, dim=0)
    return images, targets

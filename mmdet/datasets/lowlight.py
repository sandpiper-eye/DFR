from .base_det_dataset import BaseDetDataset
from mmdet.registry import DATASETS

@DATASETS.register_module()
class LowlightDataset(BaseDetDataset):
    """Unlabeled low-light image dataset."""
    METAINFO = dict(classes=(), palette=None)
    '''METAINFO = {
        'classes': ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
                    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
                    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'),
        'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192),
                    (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255),
                    (153, 69, 1), (120, 166, 157), (0, 182, 199),
                    (0, 226, 252), (182, 182, 255), (0, 0, 230), (220, 20, 60),
                    (163, 255, 0), (0, 82, 0), (3, 95, 161), (0, 80, 100),
                    (183, 130, 88)]
    }'''

    def load_data_list(self):

        import os
        img_files = [f for f in os.listdir(self.data_root) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        data_list = []
        for img in img_files:
            data_list.append({
                'img_path': os.path.join(self.data_root, img),
                'height': None,
                'width': None,
                'instances': []
            })
        return data_list

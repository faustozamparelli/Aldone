import torch

device = 'cuda' if torch.cuda.is_available() else "cpu"

best_model = models.get('yolo_nas_s',
                        num_classes=len(dataset_params['classes']),

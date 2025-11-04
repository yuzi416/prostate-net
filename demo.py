import os
import argparse
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T

from mmengine.config import Config
from pretrain.apis import ImageClassificationInferencer
from dataloader import GbRawDataset
from models import GbcNet
from segmentation import pro_crop

def mobilenet_inference(img_path, img_dir, config_file, checkpoint_file):
    inferencer = ImageClassificationInferencer(config_file, pretrained=checkpoint_file)
    result = inferencer(img_path)[0]
    pred_scores = result.get('pred_scores', None)
    if pred_scores is not None and pred_scores.size > 0:
        return int(pred_scores.argmax())
    else:
        return -1

def parse():
    parser = argparse.ArgumentParser(description='Test Model on Validation Set')
    parser.add_argument('--img_dir', required=True)
    parser.add_argument('--width', type=int, default=224)
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--meta_file', required=True)
    parser.add_argument('--load_path', required=True)
    parser.add_argument('--score_name', default="test_results.pkl")
    parser.add_argument('--mobilenet_config', required=True)
    parser.add_argument('--mobilenet_checkpoint', required=True)
    return parser.parse_args()

def main(args):
    cropped_dir = pro_crop(args.img_dir)
    transforms = T.Compose([
        T.ToPILImage(),
        T.Resize((args.width, args.height)),
        T.ToTensor()
    ])

    img_files = [f for f in os.listdir(cropped_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))]

    with open(args.meta_file, 'r') as f:
        metadata = json.load(f)

    val_dataset = GbRawDataset(cropped_dir, metadata, img_files, img_transforms=transforms)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

    model = GbcNet(num_cls=2, pretrain=False)
    model.load_state_dict(torch.load(args.load_path))
    model = model.cuda().eval()

    case_predictions = {}

    for images, targets, filenames in val_loader:
        images = images.cuda().float()
        targets = targets.cuda()

        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        pred = (probs[:,1] > 0.7).int()

        if (probs[:,1] > 0.5).all() and (probs[:,1] <= 0.7).all():
            mn_pred = mobilenet_inference(filenames[0], cropped_dir,
                                          args.mobilenet_config, args.mobilenet_checkpoint)
            if mn_pred == 1:
                pred = 1

        case_id = filenames[0].split('-')[0]
        case_data = case_predictions.setdefault(case_id, {"true":[], "preds":[], "probs":[]})
        case_data["true"].append(int(targets.item()))
        case_data["preds"].append(int(pred.item()))
        case_data["probs"].append(float(probs[:,1].item()))

    for case_id, data in case_predictions.items():
        case_true = 1 if any(t==1 for t in data["true"]) else 0
        case_pred = 1 if any(p==1 for p in data["preds"]) else 0 

    os.makedirs("output", exist_ok=True)
    out_path = os.path.join("output", args.score_name)
    with open(out_path, 'wb') as f:
        pickle.dump(case_predictions, f)

if __name__ == "__main__":
    args = parse()
    main(args)

import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as T

from dataloader import GbDataset, GbRawDataset, GbCropDataset
from models import GbcNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description='Train/Val for GBCNet')
    parser.add_argument('--img_dir', required=True, help='Directory of images')
    parser.add_argument('--set_dir', required=True, help='Directory of train/test list files')
    parser.add_argument('--train_set_name', default='train.txt')
    parser.add_argument('--test_set_name', default='test.txt')
    parser.add_argument('--meta_file', required=True, help='JSON file for metadata')
    parser.add_argument('--epochs', type=int, default=)
    parser.add_argument('--lr', type=float, default=)
    parser.add_argument('--height', type=int, default=224)
    parser.add_argument('--width', type=int, default=224)
    parser.add_argument('--no_roi', action='store_true', help='Use raw images instead of ROI crops')
    parser.add_argument('--pretrain', action='store_true', help='Use pretrained backbone')
    parser.add_argument('--load_model', action='store_true', help='Load model from checkpoint')
    parser.add_argument('--load_path', default='', help='Path to model checkpoint')
    parser.add_argument('--save_dir', default='outputs', help='Directory to save models/logs')
    parser.add_argument('--save_name', default='gbcnet', help='Base name for saved models')
    parser.add_argument('--optimizer', default='sgd')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--att_mode', default='1', help='Attention mode for model')
    return parser.parse_args()

def main(args):
    img_transforms = T.Compose([
        T.Resize((args.width, args.height)),
        T.ToTensor()
    ])

    with open(args.meta_file, 'r') as f:
        metadata = json.load(f)

    with open(os.path.join(args.set_dir, args.train_set_name), 'r') as f:
        train_labels = [line.strip() for line in f]
    with open(os.path.join(args.set_dir, args.test_set_name), 'r') as f:
        val_labels = [line.strip() for line in f]

    if args.no_roi:
        train_dataset = GbRawDataset(args.img_dir, metadata, train_labels, img_transforms=img_transforms)
        val_dataset = GbRawDataset(args.img_dir, metadata, val_labels, img_transforms=img_transforms)
    else:
        train_dataset = GbDataset(args.img_dir, metadata, train_labels, img_transforms=img_transforms)
        val_dataset = GbCropDataset(args.img_dir, metadata, val_labels, img_transforms=img_transforms)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    model = GbcNet(num_cls=2, pretrain=args.pretrain, att_mode=args.att_mode)
    model = model.to(device)

    if args.load_model and args.load_path:
        model.load_state_dict(torch.load(args.load_path, map_location=device))

    criterion = nn.CrossEntropyLoss()

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum, weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size, gamma)

    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, args.save_name + '_log.json')
    training_log = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, targets, _ in train_loader:
            images = images.to(device).float()
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, targets, _ in val_loader:
                images = images.to(device).float()
                targets = targets.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                y_true.append(targets.item())
                y_pred.append(preds.item())

        from sklearn.metrics import accuracy_score, confusion_matrix
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0

        if epoch % 10 == 0:
            save_path = os.path.join(args.save_dir, f'{args.save_name}_epoch_{epoch}.pth')
            torch.save(model.state_dict(), save_path)

        scheduler.step()

    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=4)

if __name__ == '__main__':
    args = parse_args()
    main(args)

import torch, os, gc, time
from torchvision.models import resnet34
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from pathlib import Path

from utils import epoch_time

def run_inference(config, output):
    parent = str(Path(__file__)).rsplit('\\', maxsplit=1)[0]

    datapath = os.path.join(parent, config.datapath)
    num_classes = len(os.listdir(os.path.join(parent, datapath, 'val')))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = resnet34(pretrained = False)
    model.fc = nn.Linear(512, num_classes, bias=True)
    _ = model.to(device)
    model.load_state_dict(torch.load(os.path.join(parent, 'Checkpoints', f"{config.model_name}.pth")))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    test_folder = ImageFolder(os.path.join(datapath, 'val'), transform = transform)
    test_dl = DataLoader(test_folder, batch_size = config.batch_size)

    total = 0
    correct1 = 0
    start = time.time()
    with torch.no_grad():
        for (img, label) in test_dl:
            img, label = img.to(device), label.to(device)
            predicted = model(img)
            _, predicted = torch.max(predicted.data, 1)
            total += label.size(0)
            correct1 += (predicted == label).sum().item()
            del img
            del label
            del predicted
            gc.collect()
            torch.cuda.empty_cache()
            #correct2 += torch.sum(predicted == label.data)
    print(f"Accuracy on Validation set : {correct1/total}")
    output.write(f"Accuracy on Validation set : {correct1/total}\n")
    end = time.time()
    h, m, s = epoch_time(start, end)
    print(f"Inference time : {h}hrs. {m}mins. {s}s")
    output.write(f"Inference time : {h}hrs. {m}mins. {s}s\n")
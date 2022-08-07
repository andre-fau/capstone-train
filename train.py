import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import argparse
import logging
import os
import sys
import torch.utils.data
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from smdebug import modes
    from smdebug.pytorch import get_hook
except:
    pass

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    logger.info(
        "Test set: Accuracy: {}/{} ({:.0f}%)\n".format(
            correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )
    pass

def train(model, train_loader, val_loader, criterion, optimizer, epochs):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    hook = get_hook(create_if_not_exists=True)
    if hook:
        hook.register_loss(criterion)
        
    # train the model
    for i in range(epochs):
        if hook:
            hook.set_mode(modes.TRAIN)
        model.train()
        train_loss = 0
        for _, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if hook:
            hook.set_mode(modes.EVAL)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for _, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        logger.info(
            "Epoch %d: train loss %.3f, validation loss %.3f"
            % (i, train_loss, val_loss)
        )
    return model
    
def net():
    model = models.__dict__['resnet50'](pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
        
    num_features = model.fc.in_features
    
    model.fc = nn.Sequential(
    nn.Linear(num_features, int(num_features/2)),
    nn.Linear(int(num_features/2), 5))
    return model    

def create_data_loaders(data, batch_size):
    data_transform = transforms.Compose([
        transforms.Resize((224,224)),  # Resizing following the idea of https://github.com/silverbottlep/abid_challenge
        transforms.ToTensor()
        ])
    image_dataset = torchvision.datasets.ImageFolder(root=data,transform=data_transform)
    return torch.utils.data.DataLoader(image_dataset, batch_size = batch_size)

def main(args):
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model=net()
    model.to(device)
    
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    
    train_loader = create_data_loaders(args.data_dir + '/train', args.batch_size)
    val_loader = create_data_loaders(args.data_dir + '/valid', args.batch_size)
    test_loader = create_data_loaders(args.data_dir + '/test', args.batch_size)

    model=train(model, train_loader, val_loader, loss_criterion, optimizer, args.epochs)
    
    test(model, test_loader)

    path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    #parser.add_argument('--output-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
 
    args=parser.parse_args()
    
    main(args)

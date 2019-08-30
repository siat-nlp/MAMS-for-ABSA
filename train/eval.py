import torch

def eval(model, data_loader, criterion=None):
    total_samples = 0
    correct_samples = 0
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            input0, input1, label = data
            input0, input1, label = input0.cuda(), input1.cuda(), label.cuda()
            logit = model(input0, input1)
            loss = criterion(logit, label).item() if criterion is not None else 0
            total_samples += input0.size(0)
            pred = logit.argmax(dim=1)
            correct_samples += (label == pred).long().sum().item()
            total_loss += loss * input0.size(0)
    accuracy = correct_samples / total_samples
    avg_loss = total_loss / total_samples
    if criterion is not None:
        return accuracy, avg_loss
    else:
        return accuracy
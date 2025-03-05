import torch

def eval(model, device, test_loader):
    cnt = len(test_loader.dataset)
    accuracy = 0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for X, Y in test_loader:
            X = X.to(device)
            Y = Y.to(device)

            # Forward pass: calculate predictions
            prediction = model(X)
            # label: select class that has maximum probability
            label = torch.argmax(prediction, dim=1)

            # Sum of true positives
            accuracy += torch.sum((label == Y).type(torch.FloatTensor))

        # calculte accuracy: (tp)/(number of samples)
        accuracy /= cnt

        return accuracy
from . import dataset
from .architecture import Darknet, LiteNet, MMCNN
from .dataset import DriftDataset, LidarDataset
from util.misc import bcolors
import os
import shutil
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DirectedTripletRanking(nn.Module):
    def __init__(self, margin=1):
        super(DirectedTripletRanking, self).__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        losses = torch.relu(
            negative - positive +
            (anchor - negative) ** 4 + (anchor - positive)**4 +
            self.margin
        )
        return losses.mean()


def check_accuracy(loader, model):

    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for dataentry in loader:

            ri, y = dataentry['ri'].to(device), dataentry['y'].to(device)

            pred = model(ri)
            predictions = torch.tensor(
                [torch.argmax(i) for i in pred]).to(device)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    accuracy = num_correct / num_samples
    print(
        f"Got {accuracy} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
    )
    print('Num Correct, Num Samples: ', num_correct, num_samples)
    return accuracy


def train_loop(dataloader, model, loss_fn, optimizer):
    """Train MMCNN model using train data."""

    size = len(dataloader.dataset)
    batch_count = 0
    train_loss_avg = 0
    for batch, dataentry in enumerate(dataloader):
        # Compute prediction and loss
        anchor, positive, negative = dataentry['anchor'].to(
            device), dataentry['positive'].to(device), dataentry['negative'].to(device)
        anchorval = model(anchor)
        positiveval = model(positive)
        negativeval = model(negative)
        train_loss = loss_fn(anchorval, positiveval, negativeval)
        batch_count += 1

        # Backpropagation
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        train_loss_avg += train_loss.item()
        # if batch % 100 == 0:
        #     print('Loss: ', train_loss.item())
        #     print('Vals: ')
        #     print('Anchorval', anchorval)
        #     print('Positiveval', positiveval)
        #     print('Negativeval', negativeval)
    train_loss_avg = train_loss_avg / batch_count
    print('Train Loss Average: ', train_loss_avg)
    return train_loss_avg


def test_loop(dataloader, model, loss_fn):
    """Test MMCNN model on test data."""

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss_avg = 0
    batch_count = 0

    for dataentry in dataloader:
        batch_count += 1
        anchor, positive, negative = dataentry['anchor'].to(
            device), dataentry['positive'].to(device), dataentry['negative'].to(device)
        anchorval = model(anchor)
        positiveval = model(positive)
        negativeval = model(negative)
        test_loss_avg += loss_fn(anchorval, positiveval, negativeval).item()

    test_loss_avg = test_loss_avg / batch_count
    print('Test Loss Average: ', test_loss_avg)
    return test_loss_avg


def main(args):

    torch.cuda.empty_cache()

    model_dir = args.model_dir
    print(bcolors.OKBLUE + f"\nSaving model to {model_dir}" + bcolors.ENDC)

    if os.path.isdir(model_dir):
        input('Model Dir Exists! Continue?: ')
        shutil.rmtree(model_dir)

    tb_dir = model_dir + 'tensorboard/'
    print(bcolors.OKBLUE + f"Saving tensorboard to {tb_dir}\n" + bcolors.ENDC)

    tb_writer = SummaryWriter(log_dir=tb_dir)

    if not os.path.isdir(model_dir):
        # This will also make model dir
        os.makedirs(tb_dir)

    # Hyperparams
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    epochs = args.epochs
    train_percent, test_percent = args.train_test_split[0], args.train_test_split[1]

    # Model definition
    model = LiteNet()
    model.to(device)
    loss_fn = DirectedTripletRanking()
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    dataset = DriftDataset(args.filename)

    print(bcolors.OKBLUE +
          f"\nDataset split into {int(train_percent*100)}% for training, {int(test_percent*100)}% for testing" +
          bcolors.ENDC)
    train_len, test_len = dataset.get_split_sizes(train_percent, test_percent)
    train_set, test_set = random_split(
        dataset=dataset, lengths=[train_len, test_len])

    train_dataloader = DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=True)

    for t in range(epochs):
        print(bcolors.BOLD + bcolors.OKGREEN +
              f"\nEpoch {t+1}\n-------------------------------" + bcolors.ENDC)

        train_loss = train_loop(
            dataloader=train_dataloader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer)
        # train_acc = check_accuracy(train_dataloader, model)

        test_loss = test_loop(
            dataloader=test_dataloader,
            model=model,
            loss_fn=loss_fn)
        # test_acc = check_accuracy(test_dataloader, model)

        tb_writer.add_scalar("Loss/train", train_loss, t+1)
        tb_writer.add_scalar("Loss/test", test_loss, t+1)
        # tb_writer.add_scalar("Acc/train", train_acc, t+1)
        # tb_writer.add_scalar("Acc/test", test_acc, t+1)
        tb_writer.flush()

        torch.save(model, model_dir + str(t) + '.pth')

    tb_writer.close()

    print(bcolors.OKGREEN + "Done!")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--filename",
        help="Enter directory where data is stored"
    )
    argparser.add_argument(
        "--model_dir",
        type=str,
        default='trained_models/test/',
        help="Enter directory name to store model"
    )
    argparser.add_argument(
        "--train_test_split",
        nargs='+',
        type=float,
        default=(0.8, 0.2),
        help="Enter train and test split percentages of dataset"
    )
    argparser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Enter number of epochs"
    )
    argparser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Enter learning rate"
    )
    argparser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Enter batch size"
    )
    args = argparser.parse_args()

    main(args)

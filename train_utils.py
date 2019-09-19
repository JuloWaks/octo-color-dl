import time
import matplotlib.pyplot as plt
import torch
from skimage.color import lab2rgb, rgb2lab, rgb2gray
import numpy as np
from datasets_utils import get_train_loader, get_val_loader
import os
import torch.nn.functional as F
from torch import nn


class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def __call__(self, in0, in1):
        return torch.sum((in0 - in1) ** 2, dim=1, keepdim=True)


class AverageMeter(object):
    """A handy class from the PyTorch ImageNet tutorial"""

    def __init__(self, name):
        self.reset()
        self.name = name

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def publish(self, capture_func):
        capture_func(self.name + "-avg", self.avg)
        capture_func(self.name + "-val", self.val)
        capture_func(self.name + "-count", self.count)


def to_rgb(grayscale_input, ab_input, save_path=None, save_name=None):
    """Show/save rgb image from grayscale and ab channels
     Input save_path in the form {'grayscale': '/path/', 'colorized': '/path/'}"""
    plt.clf()  # clear matplotlib
    color_image = torch.cat((grayscale_input, ab_input), 0).numpy()  # combine channels
    color_image = color_image.transpose((1, 2, 0))  # rescale for matplotlib
    color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
    color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128
    color_image = lab2rgb(color_image.astype(np.float64))
    grayscale_input = grayscale_input.squeeze().numpy()
    if save_path is not None and save_name is not None:
        plt.imsave(
            arr=grayscale_input,
            fname="{}{}".format(save_path["grayscale"], save_name),
            cmap="gray",
        )
        plt.imsave(
            arr=color_image, fname="{}{}".format(save_path["colorized"], save_name)
        )


def train(
    rank,
    args,
    model,
    device,
    dataloader_kwargs,
    train_folder="places365_standard/train",
    loss_func=nn.MSELoss,
):
    # try:
    print("starting")
    torch.manual_seed(args["seed"] + rank)

    train_loader = get_train_loader(train_folder, dataloader_kwargs, args["batch_size"])

    # optimizer = torch.optim.Adam(
    #     model.parameters(), lr=args["lr"], weight_decay=args["weight_decay"]
    # )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args["lr"],
        weight_decay=args["weight_decay"],
        momentum=0.9,
    )  # FIXME change momentum to parameter

    criterion = nn.MSELoss()
    if args["use_gpu"]:
        criterion = criterion.cuda()
    for epoch in range(1, args["epochs"] + 1):
        train_epoch(epoch, args, model, device, train_loader, optimizer, criterion)


# except Exception as err:
#     print("Handling run-time error:", err)
#     raise err


def train_epoch(epoch, args, model, device, data_loader, optimizer, criterion):

    model.train()
    pid = os.getpid()
    for batch_idx, (input_gray, input_ab, target) in enumerate(data_loader):
        if args["use_gpu"]:
            input_gray, input_ab, target = (
                input_gray.cuda(),
                input_ab.cuda(),
                target.cuda(),
            )

        optimizer.zero_grad()

        # Run forward pass
        # print("Size", input_gray.size())
        output_ab = model(input_gray)
        # print("Outside: input size", input_gray.size(), "output_size", output_ab.size())
        # print("loss: input size", input_ab.size(), "output_size", output_ab.size())

        loss = criterion(output_ab, input_ab)
        # Compute gradient and optimize
        loss.backward()
        optimizer.step()

        if batch_idx % args["log_interval"] == 0:
            print(
                "{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    pid,
                    epoch,
                    batch_idx * len(input_gray),
                    len(data_loader.dataset),
                    100.0 * batch_idx / len(data_loader),
                    loss.item(),
                )
            )


def train_epoch2(train_loader, model, criterion, optimizer, epoch, _run):
    print("Starting training epoch {}".format(epoch))
    model.train()

    # Prepare value counters and timers
    batch_time, data_time, losses = (
        AverageMeter("train.batch_time"),
        AverageMeter("train.date_time"),
        AverageMeter("train.loss"),
    )

    end = time.time()
    for i, (input_gray, input_ab, target) in enumerate(train_loader):

        # Use GPU if available

        if _run.config.get("use_gpu", False):
            input_gray, input_ab, target = (
                input_gray.cuda(),
                input_ab.cuda(),
                target.cuda(),
            )

        # Record time to load data (above)
        data_time.update(time.time() - end)

        # Run forward pass
        output_ab = model(input_gray)
        loss = criterion(output_ab, input_ab)
        losses.update(loss.item(), input_gray.size(0))

        # Compute gradient and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record time to do forward and backward passes
        batch_time.update(time.time() - end)
        end = time.time()
        losses.publish(_run)
        batch_time.publish(_run)

        # Print model accuracy -- in the code below, val refers to value, not validation
        if i % 25 == 0:
            print(
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                )
            )

    print("Finished training epoch {}".format(epoch))


def validate(val_loader, model, criterion, save_images, epoch, _run):
    model.eval()

    batch_time, data_time, losses = (
        AverageMeter("validate.batch_time"),
        AverageMeter("validate.date_time"),
        AverageMeter("validate.loss"),
    )

    end = time.time()
    already_saved_images = False
    for i, (input_gray, input_ab, target) in enumerate(val_loader):
        data_time.update(time.time() - end)

        # Use GPU
        if _run.config.get("use_gpu", False):
            input_gray, input_ab, target = (
                input_gray.cuda(),
                input_ab.cuda(),
                target.cuda(),
            )

        # Run model and record loss
        output_ab = model(input_gray)  # throw away class predictions
        loss = criterion(output_ab, input_ab)
        losses.update(loss.item(), input_gray.size(0))
        experiment_folder = _run.config.get("experiment_folder")
        # Save images to file
        if _run.config.get("save_images", False) and not already_saved_images:
            already_saved_images = True
            for j in range(min(len(output_ab), 10)):  # save at most 5 images
                save_path = {
                    "grayscale": experiment_folder + "outputs/gray/",
                    "colorized": experiment_folder + "outputs/color/",
                }
                save_name = "img-{}-epoch-{}.jpg".format(
                    i * val_loader.batch_size + j, epoch
                )
                to_rgb(
                    input_gray[j].cpu(),
                    ab_input=output_ab[j].detach().cpu(),
                    save_path=save_path,
                    save_name=save_name,
                )

        # Record time to do forward passes and save images
        batch_time.update(time.time() - end)
        end = time.time()

        # Print model accuracy -- in the code below, val refers to both value and validation
        if i % 25 == 0:
            print(
                "Validate: [{0}/{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                    i, len(val_loader), batch_time=batch_time, loss=losses
                )
            )
        losses.publish(_run)

    print("Finished validation.")
    return losses.avg


from sacred import Experiment
from sacred.observers import MongoObserver, FileStorageObserver
from time import sleep
from color_net import ColorNet

# from u_net import UNet
# from vgg_net import vggNet
from alexnet import alexnet, AlexNet
from datasets_utils import (
    GrayscaleImageFolder,
    create_dirs,
    get_train_loader,
    get_val_loader,
)
import torch
from train_utils import train, validate
from torch import nn
from os import environ
import os
import pymongo
import uuid
import torch.multiprocessing as mp
import time

_use_gpu = torch.cuda.is_available()
experiment_name = "run_big_test{}".format("_gpu" if _use_gpu else "")
ex = Experiment(experiment_name)


@ex.config
def my_config():
    recipient = "world"
    message = "Hello %s!" % recipient
    use_gpu = _use_gpu
    lr = 1e-2
    weight_decay = 0.0
    epochs = 1
    save_images = True
    run_id = uuid.uuid4()
    experiment_folder = "exp-{}/".format(run_id)
    seed = int(time.time())

    args = {
        "num_processes": 1,
        "batch_size": 64,
        "lr": lr,
        "weight_decay": weight_decay,
        "log_interval": 100,
        "use_gpu": use_gpu,
        "epochs": epochs,
        "seed": seed,
    }

    train_folder = "images/train"
    batch_size = 64
    save_exp = False


url_DB = environ["DB_URL"]
mongo_client = pymongo.MongoClient(url_DB)
mongo_obs = MongoObserver.create(client=mongo_client, db_name="octo-dl")
ex.observers.append(mongo_obs)
# ex.observers.append(FileStorageObserver("my_runs"))


@ex.automain
def my_main(
    _run,
    lr,
    weight_decay,
    message,
    use_gpu,
    epochs,
    save_images,
    experiment_folder,
    batch_size,
    save_exp,
):
    print("Epochs: {}".format(epochs))
    # args["seed"] = _run.config["seed"]

    device = torch.device("cuda" if use_gpu else "cpu")
    dataloader_kwargs = {"pin_memory": True} if use_gpu else {}
    # if save_exp:
    os.makedirs(experiment_folder + "outputs/color", exist_ok=True)
    os.makedirs(experiment_folder + "outputs/gray", exist_ok=True)
    os.makedirs(experiment_folder + "checkpoints", exist_ok=True)
    best_losses = 1e10

    seed = int(time.time())

    args = {
        "num_processes": 4,
        "batch_size": 64,
        "lr": lr,
        "weight_decay": weight_decay,
        "log_interval": 100,
        "use_gpu": use_gpu,
        "epochs": epochs,
        "seed": seed,
        "experiment_folder": experiment_folder,
    }

    train_folder = "places365_standard/train"
    val_folder = "places365_standard/val"
    trained = False
    options = dict({"num_classes": (2 * 224 * 224)})
    model = AlexNet().to(device)
    print(model)
    # model = nn.DataParallel(model)
    # model.share_memory()  # gradients are allocated lazily, so they are not shared here

    processes = []
    time1 = time.time()
    train(
        1,
        args,
        model,
        device,
        dataloader_kwargs,
        train_folder,
        nn.CrossEntropyLoss,
        val_folder,
    )
    time2 = time.time()
    print("{:s} function took {:.3f} ms".format("train", (time2 - time1) * 1000.0))

    # output_ab = model(input_gray)  # throw away class predictions
    # loss = criterion(output_ab, input_ab)
    # losses.update(loss.item(), input_gray.size(0))
    # experiment_folder = _run.config.get("experiment_folder")
    # # Save images to file

    # for rank in range(args["num_processes"]):
    #     p = mp.spawn(
    #         train,
    #         args=(rank, args, model, device, dataloader_kwargs, train_folder),
    #         nprocs=1,
    #         join=True,
    #         daemon=False,
    #     )

    #     print(p.pid)
    #     # We first train the model across `num_processes` processes
    #     p.start()
    #     print(p.pid)

    #     processes.append(p)
    # for p in processes:
    # p.join()

    # validation_loader = get_val_loader("images/val")
    # with torch.no_grad():
    #     losses = validate(validation_loader, model, criterion, save_images, epoch, _run)
    # Save checkpoint and replace old best model if current model is better
    # if losses < best_losses:
    #     best_losses = losses
    #     torch.save(
    #         model.state_dict(),
    #         experiment_folder
    #         + "checkpoints/model-epoch-{}-losses-{:.3f}.pth".format(epoch + 1, losses),
    #     )
    # Once training is complete, we can test the model
    # test(args, model, device, dataloader_kwargs)

    # print(message)
    # print("Use gpu: {}".format(use_gpu))
    # # print(_run)
    # # create_dirs()
    # model = ColorNet()
    # criterion = nn.MSELoss()
    # if use_gpu:
    #     criterion = criterion.cuda()
    #     model = model.cuda()
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # train_folder = "places365_standard/train"
    # val_folder = "places365_standard/val"
    # train_loader = get_train_loader(train_folder)
    # validation_loader = get_val_loader(val_folder)

    # for epoch in range(epochs):
    #     # Train for one epoch, then validate
    #     train(train_loader, model, criterion, optimizer, epoch, _run)


from sacred import Experiment
from sacred.observers import MongoObserver, FileStorageObserver
from time import sleep
from color_net import ColorNet
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

_use_gpu = torch.cuda.is_available()
experiment_name = "run_big_test{}".format("_gpu" if _use_gpu else "")
ex = Experiment()


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


url_DB = environ["DB_URL"]
mongo_client = pymongo.MongoClient(url_DB)
mongo_obs = MongoObserver.create(client=mongo_client, db_name="octo-dl")
ex.observers.append(mongo_obs)
# ex.observers.append(FileStorageObserver("my_runs"))


@ex.automain
def my_main(
    _run, lr, weight_decay, message, use_gpu, epochs, save_images, experiment_folder
):
    print(message)
    print("Use gpu: {}".format(use_gpu))
    # print(_run)
    # create_dirs()
    model = ColorNet()
    criterion = nn.MSELoss()
    if use_gpu:
        criterion = criterion.cuda()
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_folder = "places365_standard/train"
    val_folder = "places365_standard/train"
    train_loader = get_train_loader(train_folder)
    validation_loader = get_val_loader(val_folder)
    os.makedirs(experiment_folder + "outputs/color", exist_ok=True)
    os.makedirs(experiment_folder + "outputs/gray", exist_ok=True)
    os.makedirs(experiment_folder + "checkpoints", exist_ok=True)
    best_losses = 1e10

    print("Epochs: {}".format(epochs))

    for epoch in range(epochs):
        # Train for one epoch, then validate
        train(train_loader, model, criterion, optimizer, epoch, _run)
        with torch.no_grad():
            losses = validate(
                validation_loader, model, criterion, save_images, epoch, _run
            )
        # Save checkpoint and replace old best model if current model is better
        if losses < best_losses:
            best_losses = losses
            torch.save(
                model.state_dict(),
                experiment_folder
                + "checkpoints/model-epoch-{}-losses-{:.3f}.pth".format(
                    epoch + 1, losses
                ),
            )


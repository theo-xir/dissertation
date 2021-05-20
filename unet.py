import torch
import torch.nn as nn
import LoadData
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import argparse
from pathlib import Path
import time
import pickle
from torch.optim.optimizer import Optimizer
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os
import yaml
from argparse import Namespace
import models
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


#Code based on the code developed for the Applied Deep Learning unit coursework


#Trainer class
class Trainer:
    """
    Initialises the model
    Args: train and test data iterables, loss criterion, optimiser, summary writer, and torch device (CUDA or CPU)
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        truth_train: DataLoader,
        truth_test: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        summary_writer: SummaryWriter,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.truth_train = truth_train
        self.truth_test = truth_test
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.step = 0
        self.device = device
        self.minLoss = -1
        self.minLossEpoch = 0
        self.stop = False
    """
    Trains the network
    Args: learning rate, number of epochs, validation frequency, print frequency, log frequency
    """
    def train(
        self,
        lr:float,
        epochs: int,
        val_frequency: int,
        print_frequency: int = 10,
        log_frequency: int = 10,
    ):
        #trains the model for the number of epochs
        for epoch in range(epochs):
            self.model.train()
            data_load_start_time = time.time()
            #iterates through the data in batches
            # for batch, labels in self.train_loader:
            for i, (batch, labels) in enumerate(zip(self.train_loader, self.truth_train)):
                print("Epoch " + str(epoch) + " batch "+ str(i))
                #turns the data into tensors of the appropriate type
                batch = batch[0].to(self.device)
                labels = labels[0].to(self.device)
                data_load_end_time = time.time()

                #Compute the forward pass of the model
                logits = torch.squeeze(self.model.forward(batch, noskip=args.noskip, transconv=args.transconv))
                # print(logits.shape)
                # print(labels[0].shape)
                #Compute the loss using the MSE Loss
                loss = self.criterion(logits, labels)

                #Compute the backward pass
                loss.backward()

                #Step the optimizer and then zero out the gradient buffers.
                self.optimizer.step()
                self.optimizer.zero_grad()

                #Logging
                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, loss, data_load_time, step_time)
                if ((self.step + 1) % print_frequency) == 0:
                    self.print_metrics(epoch, loss, data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()

            #Adds information to the summary writter/logger
            self.summary_writer.add_scalar("epoch", epoch, self.step)

            #Calculates the validation loss if it is the right epoch to do so
            if ((epoch + 1) % val_frequency) == 0:
                self.validate(epoch)
                # self.validate() puts the model in validation mode, so we then switch back to train mode
                PATH = "state_dict_model.pt"
                torch.save(self.model, PATH)
                self.model.train()
            if self.stop:
                break
            

    #Prints the current step and epoch, the batch loss, time taken to load the data, and time taken to complete the step
    def print_metrics(self, epoch, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"data load time: "
                f"{data_load_time:.5f}, "
                f"step time: {step_time:.5f}"
        )
    #Adds the loss, time taken loading data, and time taken completing steps to the logs
    def log_metrics(self, epoch, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
                "loss",
                {"train": float(loss.item())},
                self.step
        )
        self.summary_writer.add_scalar(
                "time/data", data_load_time, self.step
        )
        self.summary_writer.add_scalar(
                "time/data", step_time, self.step
        )

    #Run the validation data through the network
    def validate(self, epoch):
        results = {"preds": [], "labels": [], "logits":[]}
        total_loss = 0
        self.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for i, (batch, labels) in enumerate(zip(self.val_loader, self.truth_test)):
                batch = batch[0].to(self.device)
                labels = labels[0].to(self.device)
                logits = torch.squeeze(self.model(batch,noskip=args.noskip, transconv=args.transconv))
                results["logits"].extend(logits)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()*len(batch)
                preds = logits.cpu().numpy()
                results["preds"].extend(list(preds))
                # fig, ax = plt.subplots()
                # print("should show plot now")
                # print(preds)
                #print(type(preds[0]))
                #print(type(labels[0]))
                # for j in range(batch.shape[0]):
                #     # print(j)
                #     # print(preds[j])
                #     fig, ax = plt.subplots(1,2)
                #     ax[0].contourf(preds[j],levels=51)
                #     ax[1].contourf(labels[j].cpu(),levels=51)
                #     plt.savefig("thing"+str(j)+".png")
                #     plt.clf()
        print(total_loss)
        #Store the predictions
        np.save('predictions.npy',results['preds'])

        #Calculate and print the average loss in the validation set
        average_loss = total_loss / len(self.val_loader.dataset)
        #Keeps track of the minimum loss and sets a flag to stop the training process if the loss has not decreased in a while
        if self.minLoss <0:
            self.minLoss=average_loss
            self.minLossEpoch=epoch
        elif average_loss < self.minLoss:
            self.minLoss=average_loss
            self.minLossEpoch=epoch
        elif epoch - self.minLossEpoch > 100:
            self.stop=True
        self.summary_writer.add_scalars(
                "loss",
                {"test": average_loss},
                self.step
        )
        print(f"validation loss: {average_loss:.5f}")

#Creates a unique sub directory to log this run in
def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    tb_log_dir_prefix = f'CNN_bs={args.batch_size}_lr={args.learning_rate}_momentum={args.momentum}_samples={args.samples}_run_'
    i = 0
    while i < 1000:
        tb_log_dir = Path(args.log_dir) / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)

def main(args):
    assert args.samples%10==0
    print("in main")
    #load the test and training data
    dataset, ground_truth, v = LoadData.load(args.samples, balancewind=args.balancewind, fixwind=args.fixwind, variables=args.vars, timestep=args.timestep)
    np.save('groundtruth.npy', ground_truth[int(args.samples*0.8):][0])
    global variance
    variance=v
    print("got dataset")
    print(len(dataset))
    train_dataset=TensorDataset(dataset[:int(args.samples*0.8)][0])
    test_dataset = TensorDataset(dataset[int(args.samples*0.8):][0])
    truth_train = TensorDataset(ground_truth[:int(args.samples*0.8)][0])
    truth_test = TensorDataset(ground_truth[int(args.samples*0.8):][0])
    #initialise the iterables for the training and validation sets
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        pin_memory=True,
    )
    truth_train_loader = torch.utils.data.DataLoader(
        truth_train,
        shuffle=False,
        batch_size=args.batch_size,
        pin_memory=True,
    )
    truth_test_loader = torch.utils.data.DataLoader(
        truth_test,
        shuffle=False,
        batch_size=args.batch_size,
        pin_memory=True,
    )
    # Initialises the network
    model_name=args.model
    model = getattr(models,model_name)(channels=len(args.vars),norm=args.norm, noskip=args.noskip, transconv=args.transconv)

    # Defines MSE loss criterion
    criterion = torch.nn.MSELoss()

    # Defines the Stochastic Gradient Descent optimiser with the given learning rate and momentum
    learn = float(args.learning_rate)
    print(model.parameters())
    if not args.adam:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))
    # Initiates log writting
    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )
    # Initialises the trainer class
    trainer = Trainer(
        model, train_loader, test_loader, truth_train_loader, truth_test_loader, criterion, optimizer, summary_writer, DEVICE
    )
    # Trains the model
    trainer.train(
        args.learning_rate,
        args.epochs,
        args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
    )
    # Closes the log writter
    summary_writer.close()

if __name__ == "__main__":
    print("hi")
    with open("args.yaml") as file:
        args=Namespace(**yaml.load(file,Loader=yaml.FullLoader))
        print(args)
    main(args)

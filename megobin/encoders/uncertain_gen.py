# This script has been adapted from the file available at the following address:
# https://github.com/BigDataBiology/SemiBin/blob/main/SemiBin/main.py


import os
import time
import torch
import random
import argparse
import itertools
import numpy as np
from torch import device
from functools import partial
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler


# Helper functions
def set_seed(seed: int):
    """
    Set the seed value for reproducibility
    """
    # Set the seed
    random.seed(seed)
    torch.manual_seed(seed)


def log1mexp(x: torch.Tensor):
    """
    Compute `log(1 - exp(-x))` in a numerically stable way for x > 0
    """
    log2 = torch.log(torch.tensor(2.0, dtype=x.dtype, device=x.device))
    return torch.where(
        x < log2, torch.log(-torch.expm1(-x)), torch.log1p(-torch.exp(-x))
    )


class KmerConverter:
    def __init__(self, k):

        # Define the letters, k-mer size, and the base complement
        self.__k = k
        self.__letters = ["A", "C", "G", "T"]
        self.__kmer2id = {
            "".join(kmer): i
            for i, kmer in enumerate(itertools.product(self.__letters, repeat=self.__k))
        }
        self.__kmers_num = len(self.__kmer2id)

    def get_k(self):
        return self.__k

    def get_kmers_num(self):
        return self.__kmers_num

    def seq2kmer(self, seq: str, normalized: bool = True):

        # Get the k-mer profile
        kmer2id = [
            self.__kmer2id[seq[i : i + self.__k]]
            for i in range(len(seq) - self.__k + 1)
        ]
        kmer_profile = np.bincount(kmer2id, minlength=self.__kmers_num)

        if normalized:
            kmer_profile = kmer_profile / kmer_profile.sum()

        return kmer_profile


class PairDataset(Dataset):
    def __init__(
        self,
        file_path,
        k,
        neg_sample_per_pos=1000,
        max_seq_num=0,
        input_type="dna",
        verbose=True,
        seed=0,
    ):
        """
        PairDataset constructor
        """
        # Set the parameters
        self.__kmers = None  # A tensor of shape N x D, storing the k-mer profiles
        self.__indices = None  # A tensor of shape L x 2, storing the indices of the positive and negative pairs
        self.__labels = None  # A tensor of shape L, storing the labels of the pairs
        self.__kc = KmerConverter(k)
        self.__neg_sample_per_pos = neg_sample_per_pos

        # Set the seed
        set_seed(seed)

        if verbose:
            print(f"+ Reading the data file.")
            print(f"\t- File path: {file_path}")
            init_time = time.time()

        # Get the number of lines
        num_of_lines = sum(1 for _ in open(file_path, "r"))

        # If the max_read_num is set, then sample the line numbers to read
        if max_seq_num > 0:
            chosen_lines = random.sample(range(num_of_lines), max_seq_num)
            chosen_lines.sort()
        # Otherwise, read all the lines
        else:
            chosen_lines = list(range(num_of_lines))

        # Read the file
        chosen_line_idx = 0
        left_kmer_profiles, right_kmer_profiles = [], []
        f = open(file_path, "r")
        for line_no, line in enumerate(f):
            if line_no == chosen_lines[chosen_line_idx]:
                left_seq, right_seq = line.strip().split(",")

                left_kmer_profiles.append(
                    np.asarray(list(map(float, left_seq.split())))
                    if input_type.lower() == "kmer"
                    else self.__kc.seq2kmer(left_seq)
                )
                right_kmer_profiles.append(
                    np.asarray(list(map(float, right_seq.split())))
                    if input_type.lower() == "kmer"
                    else self.__kc.seq2kmer(right_seq)
                )

                # Increment the chosen_line_idx
                chosen_line_idx += 1
                # If all the chosen lines have been read, then break
                if chosen_line_idx >= len(chosen_lines):
                    break
        f.close()

        # Combine the left and right k-mer profiles.
        # The first half of the profiles are the left ones and the second half are the right ones
        self.__kmers = torch.from_numpy(
            np.asarray(left_kmer_profiles + right_kmer_profiles)
        ).to(torch.float)

        if verbose:
            print(f"\t- Completed in {time.time() - init_time:.2f} seconds.")
            # print the time information in 2 decimal points
            print(f"\t- The input file contains: {num_of_lines} sequence pairs.")
            print(f"\t- For training, {max_seq_num} sequence pairs will be used.")

        # Construct the indices storing the positive and negative pairs
        if verbose:
            print(f"+ Constructing the indices for positive and negative sample pairs")
            init_time = time.time()

        # Sample 'pos_indices' positive pair indices
        pos_indices = torch.vstack(
            (
                torch.arange(self.__kmers.shape[0] // 2, dtype=torch.long),
                torch.arange(self.__kmers.shape[0] // 2, dtype=torch.long)
                + self.__kmers.shape[0] // 2,
            )
        )
        # Sample 'neg_sample_per_pos' negative pair indices for each positive pair
        temp_weights = torch.ones((self.__kmers.shape[0],), dtype=torch.float)
        neg_indices = torch.vstack(
            (
                torch.multinomial(
                    temp_weights,
                    pos_indices.shape[1] * self.__neg_sample_per_pos,
                    replacement=True,
                ),
                torch.multinomial(
                    temp_weights,
                    pos_indices.shape[1] * self.__neg_sample_per_pos,
                    replacement=True,
                ),
            )
        )
        # Concatenate the positive and negative indices
        self.__indices = torch.hstack((pos_indices, neg_indices))
        self.__labels = torch.hstack(
            (
                torch.ones((pos_indices.shape[1],), dtype=torch.float),
                torch.zeros((neg_indices.shape[1],), dtype=torch.float),
            )
        )

        if verbose:
            print(f"\t- Completed in {time.time() - init_time:.2f} seconds.")
            print(
                f"\t- Training dataset contains {pos_indices.shape[1]} positive pairs."
            )
            print(
                f"\t- Training dataset contains {neg_indices.shape[1]} negative pairs."
            )

    def __len__(self):
        """
        Return the number of pairs
        """
        return self.__indices.shape[1]

    def __getitem__(self, idx):

        return (
            torch.index_select(self.__kmers, dim=0, index=self.__indices[0, idx]),
            torch.index_select(self.__kmers, dim=0, index=self.__indices[1, idx]),
            self.__labels[idx],
        )


class Model(torch.nn.Module):
    def __init__(self, k: int, out_dim: int = 256, seed: int = 0):
        """
        Initialize the VIB model
        """
        super(Model, self).__init__()

        # Set the parameters
        self.__k = k
        self.__out_dim = out_dim
        # Define the k-mer converter object
        self.__kc = KmerConverter(self.__k)
        # Set the seed
        set_seed(seed)

        # Define the layers
        self.mean = torch.nn.Sequential(
            torch.nn.Linear(self.__kc.get_kmers_num(), 512, dtype=torch.float),
            torch.nn.BatchNorm1d(512, dtype=torch.float),
            torch.nn.Sigmoid(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, self.__out_dim, dtype=torch.float),
        )

        self.std = torch.nn.Sequential(
            torch.nn.Linear(self.__kc.get_kmers_num(), 512, dtype=torch.float),
            torch.nn.BatchNorm1d(512, dtype=torch.float),
            torch.nn.Sigmoid(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, self.__out_dim, dtype=torch.float),
        )

    def encoder(self, kmers: torch.Tensor):
        """
        For given kmers, return the mean and std
        """
        mean = self.mean(kmers)

        log_cov = self.std(kmers)
        cov = torch.exp(log_cov)

        return mean, cov

    def forward(self, left_kmers: torch.Tensor, right_kmers: torch.Tensor):
        """
        Forward pass
        """
        left_mean, left_cov = self.encoder(left_kmers)
        right_mean, right_cov = self.encoder(right_kmers)

        return left_mean, left_cov, right_mean, right_cov

    def get_k(self):
        """
        Return the k value
        """
        return self.__k

    def get_out_dim(self):
        """
        Return the output dimension
        """
        return self.__out_dim

    def get_seed(self):
        """
        Return the seed value
        """
        return self.__seed

    def seq2emb(self, sequences: list, normalized: bool = True, input_type="dna"):
        """
        Get the embeddings for the given list of DNA sequences
        """

        if input_type.lower() == "dna":
            kmers = torch.from_numpy(
                np.asarray(
                    [
                        self.__kc.seq2kmer(seq, normalized=normalized)
                        for seq in sequences
                    ]
                )
            ).to(torch.float)

        elif input_type.lower() == "kmer":
            kmers = torch.from_numpy(np.asarray(sequences)).to(torch.float)

        else:
            raise ValueError(f"Unknown input type: {input_type}")

        with torch.no_grad():
            self.eval()

            if normalized:
                kmers = kmers / kmers.sum(dim=1, keepdim=True)

            means, covs = self.encoder(kmers)

        return means.detach().numpy(), covs.detach().numpy()

    def save(self, path: str):
        """
        Save the model
        """
        torch.save(
            [{"k": self.get_k(), "out_dim": self.get_out_dim()}, self.state_dict()],
            path,
        )


def loss_func(
    left_mean: torch.Tensor,
    left_cov: torch.Tensor,
    right_mean: torch.Tensor,
    right_cov: torch.Tensor,
    labels: torch.Tensor,
    threshold=1.0,
    loss_name: str = "mahalanobis",
    include_std: bool = True,
):
    """
    Definition of the loss functions
    """

    if loss_name == "l2":
        left_mean = left_mean
        right_mean = right_mean

        squared_dist = torch.norm(left_mean - right_mean, dim=1) ** 2
        output = labels * squared_dist - (1 - labels) * torch.log(
            1 - torch.exp(-squared_dist) + 1e-6
        )
        return output.mean()

    elif loss_name == "mahalanobis":
        # Compute the term (m_i - m_j)^2
        squared_diff = (left_mean - right_mean) ** 2

        if not include_std:
            # Compute the log expectation
            log_expectation = -squared_diff.sum(dim=1) / 4.0

        else:
            # Compute the term (m_i - m_j)^2 / (s_i + s_j)
            squared_diff = squared_diff / (left_cov + right_cov)

            # Compute the log expectation
            log_expectation = -squared_diff.sum(dim=1) / 1.0

        output = -labels * log_expectation - (1 - labels) * torch.log(
            1
            - torch.exp(torch.nn.functional.threshold(log_expectation, -threshold, 0))
            + 1e-6
        )
        return output.mean()

    else:
        raise ValueError(f"Unknown loss function name: {loss_name}")


def train_batch(
    model,
    criterion,
    optimizer,
    left_kmers: torch.Tensor,
    right_kmers: torch.Tensor,
    labels: torch.Tensor,
):
    # Zero your gradients since PyTorch accumulates gradients on subsequent backward passes.
    optimizer.zero_grad()

    # Make predictions for the current epoch
    left_mean, left_cov, right_mean, right_cov = model(left_kmers, right_kmers)

    # Compute the loss and backpropagate
    batch_loss = criterion(left_mean, left_cov, right_mean, right_cov, labels)
    batch_loss.backward()

    # Update the model parameters
    optimizer.step()

    return batch_loss


def train_single_epoch(device, model, criterion, optimizer, data_loader):

    epoch_loss = 0.0
    for data in data_loader:
        left_kmers, right_kmers, labels = data
        left_kmers = left_kmers.reshape(-1, left_kmers.shape[-1]).to(device)
        right_kmers = right_kmers.reshape(-1, right_kmers.shape[-1]).to(device)
        labels = labels.reshape(-1).to(device)

        # Run the training for the current batch
        batch_loss = train_batch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            left_kmers=left_kmers,
            right_kmers=right_kmers,
            labels=labels,
        )

        # Get the epoch loss for reporting
        epoch_loss += batch_loss

    # Get the average epoch loss
    average_epoch_loss = epoch_loss.item() / len(data_loader)

    return average_epoch_loss


def train(
    device,
    distributed,
    model,
    strategy,
    lr,
    threshold,
    loss_name,
    data_loader,
    epoch_num: int,
    save_every: int,
    output_path,
    summary_writer,
):
    # Check the strategy name
    assert strategy in ["sequential", "mean_only", "var_only", "together"], (
        f"Unknown strategy name: {strategy}"
    )

    ### Define the optimizer
    # At the beginning, we will not train the standard deviation parameters if the strategy is "sequential" or "mean_only"
    include_std = False if strategy in ["sequential", "mean_only"] else True
    # Get the model parameters to train
    model_params = []
    for name, param in model.named_parameters():
        if "std" in name and (strategy in ["sequential", "mean_only"]):
            param.requires_grad = False
        elif "mean" in name and strategy in ["var_only"]:
            param.requires_grad = False
        else:
            model_params.append(param)

    ### Define the loss function.
    # Since we won't train the std weights at the beginning, we will not include them in the loss function
    criterion = partial(
        loss_func, threshold=threshold, loss_name=loss_name, include_std=include_std
    )
    # Define the optimizer
    optimizer = torch.optim.Adam(model_params, lr=lr)

    ### Start training
    print(f"+ Training started.")

    # Set the correct training/evaluation modes
    if strategy in ["mean_only", "sequential"]:
        model.mean.train()
        model.std.eval()
    elif strategy == "var_only":
        model.std.train()
        model.mean.eval()
    elif strategy == "together":
        model.mean.train()
        model.std.train()
    else:
        raise ValueError(f"Unknown strategy name: {strategy}")

    for current_epoch in range(epoch_num):
        # Set the epoch for the sampler
        if distributed:
            data_loader.sampler.set_epoch(current_epoch)

        # For the second half of the training, we will also train the std weights if the strategy is "sequential"
        if (
            (current_epoch + 1) > epoch_num * (1.0 / 2)
            and include_std == False
            and strategy == "sequential"
        ):
            # # Set the gradient of the std weights to True
            std_params = []
            for name, param in model.named_parameters():
                param.requires_grad = False
                if "std" in name:
                    param.requires_grad = True
                    std_params.append(param)

            # Set the include_std to True
            include_std = True
            # Define the optimizer
            optimizer = torch.optim.Adam(std_params, lr=lr)
            # Define the loss function
            criterion = partial(
                loss_func,
                threshold=threshold,
                loss_name=loss_name,
                include_std=include_std,
            )
            if (distributed and device == 0) or (not distributed):
                print(
                    f"   - Only the covariance parameters are included in the optimizer now! (lr = {lr})"
                )

            # Set the correct model modes
            model.mean.eval()
            model.std.train()

        ## Run the training for the current epoch
        init_time = time.time()
        loss = train_single_epoch(
            device=device,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            data_loader=data_loader,
        )
        loss_time = time.time() - init_time
        if distributed:
            if device == 0:
                if summary_writer is not None:
                    summary_writer.add_scalar("Loss/train", loss, current_epoch)
                    summary_writer.add_scalar("Time/epoch", loss_time, current_epoch)
                print(
                    f"\t- Epoch: {current_epoch + 1}/{epoch_num} - Loss: {loss} ({loss_time:.2f} secs)"
                )
        else:
            if summary_writer is not None:
                summary_writer.add_scalar("Loss/train", loss, current_epoch)
                summary_writer.add_scalar("Time/epoch", loss_time, current_epoch)
            print(
                f"\t- Epoch: {current_epoch + 1}/{epoch_num} - Loss: {loss} ({loss_time:.2f} secs)"
            )

        ## Save the checkpoint if necessary
        if save_every > 0 and (current_epoch + 1) % save_every == 0:
            # Get the folder path of the output file and define the checkpoint path
            checkpoint_path = os.path.join(
                output_path + f".epoch_{current_epoch + 1}.checkpoint"
            )
            if distributed:
                if device == 0:
                    # Get the target folder path and if it does not exist, create it
                    if not os.path.exists(os.path.dirname(output_path)):
                        os.makedirs(os.path.dirname(output_path))

                    model.module.save(checkpoint_path)
            else:
                # Get the target folder path and if it does not exist, create it
                if not os.path.exists(os.path.dirname(output_path)):
                    os.makedirs(os.path.dirname(output_path))

                model.save(checkpoint_path)

    if distributed:
        if device == 0:
            model.module.save(output_path)
            print(f"Model saved to {output_path}")
    else:
        model.save(output_path)
        print(f"+ Model saved to {output_path}")
    print(f"\t- Completed (Device {device}).")


def main_worker(
    device,
    world_size,
    distributed: bool,
    training_dataset: PairDataset,
    output_path: str,
    checkpoint: str,
    k: int,
    out_dim: int,
    strategy: str,
    lr: float,
    epoch_num: int,
    batch_size: int,
    threshold: float,
    workers_num: int,
    save_every: int,
    loss_name: str,
    log_dir: str,
    seed: int,
):
    ### Initialize the device
    if distributed:
        # Set the environment variables for distributed training if not already set
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "12355"
        print(
            f"+ Device: {device} | MASTER_ADDR: {os.environ['MASTER_ADDR']} | MASTER_PORT: {os.environ['MASTER_PORT']}"
        )
        torch.cuda.set_device(device)
        torch.distributed.init_process_group(
            backend="nccl", rank=device, world_size=world_size
        )

    # Define a DataLoader that iterates through batches of data.
    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        num_workers=workers_num,
        pin_memory=True,
        sampler=DistributedSampler(training_dataset) if distributed else None,
        shuffle=False if distributed else True,
    )

    ### Define the model
    model = Model(k=k, out_dim=out_dim, seed=seed)
    if checkpoint != "":
        print(f"+ Loading the model from the checkpoint: {checkpoint}")
        # Load the model from the checkpoint
        kwargs, model_state_dict = torch.load(
            checkpoint, map_location=torch.device("cpu")
        )
        model = Model(**kwargs)
        model.load_state_dict(model_state_dict)

    # Move the model to the device
    if distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(
            model.to(f"cuda:{device}"), device_ids=[device], find_unused_parameters=True
        )
    else:
        model.to(device)

    ### Define the log writer
    summary_writer = None
    if log_dir != "":
        summary_writer = SummaryWriter(log_dir=log_dir)

    ### Train the model
    train(
        device=device,
        distributed=distributed,
        model=model,
        strategy=strategy,
        lr=lr,
        threshold=threshold,
        loss_name=loss_name,
        data_loader=training_loader,
        epoch_num=epoch_num,
        save_every=save_every,
        output_path=output_path,
        summary_writer=summary_writer,
    )

    # Close the summary writer
    if summary_writer is not None:
        summary_writer.close()

    # Terminate the processes
    if distributed:
        # Wait for all processes to complete
        torch.distributed.barrier()
        # Release the distributed training resources
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument(
        "--input", type=str, help="Input DNA sequences file", required=True
    )
    parser.add_argument(
        "--input_type",
        type=str,
        help="Input file type (dna, kmer)",
        required=False,
        default="dna",
    )
    parser.add_argument(
        "--output", type=str, help="Model file path to save", required=True
    )
    parser.add_argument(
        "--checkpoint", type=str, default="", help="Checkpoint file path to load"
    )
    parser.add_argument("--k", type=int, default=4, help="k value")
    parser.add_argument(
        "--out_dim", type=int, default=256, help="Output dimension value"
    )
    parser.add_argument(
        "--neg_sample_per_pos",
        type=int,
        default=200,
        help="Negative samples count per positive",
    )
    parser.add_argument(
        "--max_seq_num",
        type=int,
        default=100000,
        help="Maximum number of sequences to get from the file",
    )
    parser.add_argument("--epoch", type=int, default=50, help="Epoch number")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--threshold", type=float, default=1.0, help="Threshold value")
    parser.add_argument(
        "--strategy",
        type=str,
        default="sequential",
        help="Learning starting strategy (sequential, mean_only, var_only, together)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=100000, help="Batch size (0: no batch)"
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=5,
        help="Save checkpoint every x epochs (0 for no checkpoint)",
    )
    parser.add_argument(
        "--distributed",
        type=int,
        default=0,
        help="Use distributed training (0: no, 1: yes)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to use (cpu, mps or gpu)"
    )
    parser.add_argument(
        "--workers", type=int, default=2, help="Number of data loading workers"
    )
    parser.add_argument(
        "--loss_name",
        type=str,
        default="mahalanobis",
        help="Loss function (l2, mahalanobis)",
    )
    parser.add_argument(
        "--log_dir", type=str, default="", help="Path directory to save the log file"
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="Seed for random number generator"
    )

    args = parser.parse_args()

    # Define the world size, i.e. total number of processes participating in the distributed training job
    world_size = None
    if args.distributed:
        assert torch.cuda.is_available(), "Distributed training requires CUDA"
        nodes_num = 1
        world_size = torch.cuda.device_count() * nodes_num
    else:
        if args.device == "gpu":
            assert torch.cuda.is_available(), "GPU is not available"
            device = torch.device(f"cuda")
        elif args.device == "mps":
            assert torch.backends.mps.is_available(), "MPS is not available"
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    print(f"+ Information")
    # Get the model name from the output path
    print(f"\t- k value: {args.k}")
    print(f"\t- Loss name: {args.loss_name}")
    print(f"\t- Epoch number: {args.epoch}")
    print(f"\t- Batch size: {args.batch_size}")
    print(f"\t- Learning rate: {args.lr}")
    print(f"\t- Threshold value: {args.threshold}")
    print(f"\t- Output dimension: {args.out_dim}")
    print(f"\t- Negative sample per positive pair: {args.neg_sample_per_pos}")
    print(f"\t- Maximum number of sequence pairs to use: {args.max_seq_num}")
    if args.checkpoint != "":
        print(f"\t- Checkpoint file: {args.checkpoint}")
    print(f"\t- Learning starting strategy: {args.strategy}")
    if args.distributed:
        print(f"\t- Distributed training is activated with {world_size} GPUs")
    else:
        print(f"\t- No distributed training")
        print(f"\t- Device: {device}")
    print(f"\t- Number of data loading workers: {args.workers}")
    print(f"\t- Seed value: {args.seed}")
    print(f"\t- Output model name: {os.path.basename(args.output)}")

    ### Read the dataset and construct the data loader
    training_dataset = PairDataset(
        file_path=args.input,
        k=args.k,
        neg_sample_per_pos=args.neg_sample_per_pos,
        input_type=args.input_type,
        max_seq_num=args.max_seq_num,
        seed=args.seed,
    )
    # Define the arguments for the main worker function
    arguments = (
        world_size,
        args.distributed,
        training_dataset,
        args.output,
        args.checkpoint,
        args.k,
        args.out_dim,
        args.strategy,
        args.lr,
        args.epoch,
        args.batch_size,
        args.threshold,
        args.workers,
        args.save_every,
        args.loss_name,
        args.log_dir,
        args.seed,
    )
    if args.distributed:
        torch.multiprocessing.spawn(
            main_worker, nprocs=world_size, join=True, args=arguments
        )
    else:
        main_worker(*((device,) + arguments))
    print("+ Completed.")

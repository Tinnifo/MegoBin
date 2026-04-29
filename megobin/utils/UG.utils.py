# This script has been adapted from the file available at the following address:
# https://github.com/abdcelikkanat/kum/blob/icml_final/evaluation/utils.py

import numpy as np
import transformers
import torch
import torch.utils.data as util_data
import torch.nn as nn
import tqdm
import os
import time
import collections
from scipy.spatial import distance
import itertools
from sklearn.preprocessing import normalize
from src.model import Model
from scipy.optimize import linear_sum_assignment

# # Here, we define the batch sizes for inference. It might be important for the memory requirements of the model.
# MODEL2BATCH_SIZE = {
#     "tnf": 100, "tnf_k": 100, "hyenadna": 100, "dnabert2": 20, "nt": 64, "dnaberts": 20,
# }


def clean_by_variance(dna_sequences, model_path, filter_ratio):

    # If filter ratio is less than 0, then randomly select the sequences
    if filter_ratio < 0:
        print(
            f"Filter ratio {filter_ratio} is negative, so randomly select the sequences!"
        )
        chosen_indices = np.random.choice(
            len(dna_sequences),
            size=int((1 + filter_ratio) * len(dna_sequences)),
            replace=False,
        )
        return chosen_indices

    # Filter the sequences that have the largest minimum covariances
    kwargs, model_state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model = Model(**kwargs)
    model.load_state_dict(model_state_dict)
    # Get the mean and stds
    _, covs = model.seq2emb(dna_sequences)

    # Get the sorted indices of the stds
    sorted_idx = np.argsort(np.max(covs, axis=1))
    selected_idx = sorted_idx[: int((1 - filter_ratio) * len(covs))]
    print(
        f"The data had {len(sorted_idx)} sequences and {len(selected_idx)} remained after cleaning by variances!"
    )

    return selected_idx


def convert_labels2int(labels):

    # Convert the sequence labels to numeric values
    label2id = {l: i for i, l in enumerate(set(labels))}
    int_labels = np.array([label2id[l] for l in labels])

    return int_labels


def filter_sequences(data_path, shorten, min_len=0, abundance=0):

    import csv

    with open(data_path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        data = list(reader)[1:]

    dna_sequences = [d[0] for d in data]
    labels = [d[1] for d in data]

    if shorten:
        # Shorten the sequences if they are longer than the MAX_SEQ_LEN value
        dna_sequences = [seq[:shorten] for seq in dna_sequences]

    if min_len:
        # Filter sequences with length 'min_len'
        filterd_idx = [i for i, seq in enumerate(dna_sequences) if len(seq) >= min_len]
        dna_sequences = [dna_sequences[i] for i in filterd_idx]
        labels = [labels[i] for i in filterd_idx]

    if abundance:
        # Filter sequences with low abundance labels (less than 'abundance')
        label_counts = collections.Counter(labels)
        filterd_idx = [i for i, l in enumerate(labels) if label_counts[l] >= abundance]
        dna_sequences = [dna_sequences[i] for i in filterd_idx]
        labels = [labels[i] for i in filterd_idx]

    print(
        f"+ After filtering, there are {len(dna_sequences)} sequences from {len(set(labels))} classes."
    )

    return dna_sequences, labels


def log1mexp(x):
    log2 = np.log(2)
    return np.where(x < log2, np.log(-np.expm1(-x)), np.log1p(-np.exp(-x)))


def get_embedding(
    dna_sequences,
    model_name,
    species,
    sample_id,
    k=4,
    model_path=None,
    embedding_file_path=None,
):
    """
    Get the embeddings (and covariances depending on the method) of the DNA sequences using the specified model.

    """

    # Define the batch size to get the embeddings
    # This is for inference only, not training and it is important for the memory requirements
    batch_size = 256  # MODEL2BATCH_SIZE[model_name]

    # Load the embedding file if it exits
    if os.path.exists(embedding_file_path):
        print(f"\t- Load embeddings from file {embedding_file_path}")
        output = np.load(embedding_file_path)
        if output.ndim == 3:
            # If the output is a tuple of embedding and covariances
            output = (output[0], output[1])

    else:
        print(
            f"\t- Calculate embedding for the {species} {sample_id} data by {model_name} model"
        )

        if model_name == "tnf":
            embedding = calculate_tnf(dna_sequences)
            output = normalize(embedding, norm="l2")

        elif model_name == "tnf_k":
            embedding = calculate_tnf(dna_sequences, kernel=True)
            output = normalize(embedding, norm="l2")

        elif model_name == "hyenadna":
            embedding = calculate_llm_embedding(
                dna_sequences,
                model_name_or_path="LongSafari/hyenadna-medium-450k-seqlen-hf",
                model_max_length=20000,
                batch_size=batch_size,
            )
            output = normalize(embedding, norm="l2")

        elif model_name == "dnabert2":
            embedding = calculate_llm_embedding(
                dna_sequences,
                model_name_or_path="zhihan1996/DNABERT-2-117M",
                model_max_length=5000,
                batch_size=batch_size,
            )
            output = normalize(embedding, norm="l2")

        elif model_name == "nt":
            embedding = calculate_llm_embedding(
                dna_sequences,
                model_name_or_path="InstaDeepAI/nucleotide-transformer-v2-100m-multi-species",
                model_max_length=2048,
                batch_size=batch_size,
            )
            output = normalize(embedding, norm="l2")

        elif model_name == "dnaberts":
            embedding = calculate_llm_embedding(
                dna_sequences,
                model_name_or_path="zhihan1996/DNABERT-S",
                model_max_length=5000,
                batch_size=batch_size,
            )
            output = normalize(embedding, norm="l2")

        elif model_name == "kmerprofile":
            norm = "l1"
            embedding = calculate_tnf(dna_sequences, k=k)
            output = normalize(embedding, norm=norm)

        elif model_name == "ours":
            kwargs, model_state_dict = torch.load(
                model_path, map_location=torch.device("cpu")
            )
            model = Model(**kwargs)
            model.load_state_dict(model_state_dict)
            embedding, cov = model.seq2emb(dna_sequences)
            # Note that the output is a tuple of embedding and covariances
            output = (embedding, cov)

        else:
            raise ValueError(f"Unknown model {model_name}")

        # Save the embedding file if a valid path is provided
        if embedding_file_path != "":
            # Get the directory of the embedding file
            embedding_file_dir = os.path.dirname(embedding_file_path)
            # Save the embedding file
            os.makedirs(embedding_file_dir, exist_ok=True)
            with open(embedding_file_path, "wb") as f:
                np.save(f, output)

    return output


def compute_class_center_medium_similarity(
    features, labels, metric="dot", chunk_size=512
):

    # Sort the embeddings (and covariances if exists) by labels
    idx = np.argsort(labels)
    labels = labels[idx]
    if isinstance(features, tuple):
        embeddings, covs = features[0][idx], features[1][idx]
    else:
        embeddings, covs = features[idx], None

    # Get the counts of samples per class, i.e. the number of samples per class
    n_sample_per_class = np.bincount(labels)

    # We will compute the similarities between the class centers and the samples. Each entry in the all_similarities
    # array will store the similarity between the corresponding embedding and its class center it belongs to.
    all_similarities = np.zeros(len(embeddings))

    # Iterate over each class
    count = 0
    for i in range(len(n_sample_per_class)):
        # Get the start and end indices of the embeddings located at the class i
        start = count
        end = count + n_sample_per_class[i]

        # Get the mean of the embedding vectors located at the start:end indices
        features1 = np.mean(embeddings[start:end], axis=0, keepdims=True)
        # Define the second set of features as the embeddings located at the start:end indices
        features2 = embeddings[start:end]

        # If covariances are provided, we also need to get the mean of the covariances
        if covs is not None:
            # features1 = (features1, np.mean(covs[start:end], axis=0, keepdims=True))
            features1 = (features1, 1e0 * np.ones_like(features1))
            features2 = (features2, covs[start:end])

        # Compute the similarity matrix between the two sets of features
        similarities = compute_similarity_matrix(
            features1=features1,
            features2=features2,
            metric=metric,
            chunk_size=chunk_size,
        ).reshape(-1)

        # Store the similarities in the all_similarities array
        all_similarities[start:end] = similarities

        count += n_sample_per_class[i]

    all_similarities.sort()
    percentile_values = []
    for percentile in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        value = all_similarities[int(percentile / 100 * len(embeddings))]
        percentile_values.append(value)
    print(f"+ Percentile values: {np.asarray(percentile_values)}")

    return percentile_values


def KMedoid(
    features,
    min_similarity,
    min_bin_size=100,
    max_iter=300,
    metric="dot",
    chunk_size=512,
):

    # Compute the similarities between the features
    similarities = compute_similarity_matrix(
        features, metric=metric, chunk_size=chunk_size
    )

    # Normalization might be important if similarities include negative values
    small_number = 0  # similarities.min()
    similarities = similarities - small_number
    min_similarity = min_similarity - small_number

    # Set the values below min_similarity to 0
    similarities[similarities < min_similarity] = 0

    if isinstance(features, tuple):
        p = -np.ones(len(features[0]), dtype=float)
    else:
        p = -np.ones(len(features), dtype=float)

    row_sum = np.sum(similarities, axis=1)
    iter_count = 1
    while np.any(p == -1):
        if iter_count == max_iter:
            break
        # print(f"Iteration {iter_count} with {np.sum(p == -1)} unassigned elements")

        # Select the seed index, i.e. medoid index (Line 4)
        s = np.argmax(row_sum)
        # Initialize the current medoid (Line 4)
        # current_medoid = (features[0][s], features[1][s]) if isinstance(features, tuple) else features[s]
        current_medoid = (
            (features[0][s], 1e0 * np.ones_like(features[1][s]))
            if isinstance(features, tuple)
            else features[s]
        )

        selected_idx = None
        # Optimize the current medoid (Line 5-8)
        for t in range(3):
            # For the current medoid, find its similarities
            if isinstance(features, tuple):
                features2 = (
                    np.expand_dims(current_medoid[0], axis=0),
                    np.expand_dims(current_medoid[1], axis=0),
                )
            else:
                features2 = np.expand_dims(current_medoid, axis=0)
            similarity = compute_similarity_matrix(
                features1=features,
                features2=features2,
                metric=metric,
                chunk_size=chunk_size,
            ).squeeze()
            similarity = similarity - small_number

            # Determine the indices that are within the similarity threshold
            idx_within = similarity >= min_similarity
            # Determine the available indices, i.e. the indices that have not been assigned to a cluster yet
            idx_available = p == -1
            # Get the indices that are both within the similarity threshold and available
            selected_idx = np.where(np.logical_and(idx_within, idx_available))[0]
            # Determine the new k-medoid
            if isinstance(features, tuple):
                # current_medoid = (np.mean(features[0][selected_idx], axis=0), np.mean(features[1][selected_idx], axis=0))
                current_medoid = (
                    np.mean(features[0][selected_idx], axis=0),
                    1e0 * np.ones(shape=(1, features[1].shape[1])),
                )
            else:
                current_medoid = np.mean(features[selected_idx], axis=0)

        # Assign the cluster labels and update the row sums (Lines 9-10)
        if selected_idx is not None:
            p[selected_idx] = iter_count
            row_sum -= np.sum(similarities[:, selected_idx], axis=1)
            row_sum[selected_idx] = 0
            # print(f"Current label: {iter_count}, Number of assigned elements: {len(selected_idx)}")
        else:
            raise ValueError("No selected index")

        iter_count += 1

    # remove bins that are too small
    unique, counts = np.unique(p, return_counts=True)
    for label, count in zip(unique, counts):
        if count < min_bin_size:
            p[p == label] = -1

    return p


def calculate_tnf(dna_sequences, kernel=False, k=4):
    # Define all possible tetra-nucleotides
    nucleotides = ["A", "T", "C", "G"]

    multi_nucleotides = [
        "".join(kmer) for kmer in itertools.product(nucleotides, repeat=k)
    ]

    # build mapping from multi-nucleotide to index
    tnf_index = {tn: i for i, tn in enumerate(multi_nucleotides)}

    # Iterate over each sequence and update counts
    embedding = np.zeros((len(dna_sequences), len(multi_nucleotides)))
    for j, seq in enumerate(dna_sequences):
        for i in range(len(seq) - k + 1):
            multi_nuc = seq[i : i + k]
            embedding[j, tnf_index[multi_nuc]] += 1

    if kernel:
        raise ValueError("Not Implemented!")

    return embedding


def calculate_llm_embedding(
    dna_sequences, model_name_or_path, model_max_length=400, batch_size=20
):

    # reorder the sequences by length
    lengths = [len(seq) for seq in dna_sequences]
    idx = np.argsort(lengths)
    dna_sequences = [dna_sequences[i] for i in idx]
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=None,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    is_hyenadna = "hyenadna" in model_name_or_path
    is_nt = "nucleotide-transformer" in model_name_or_path
    is_dnabert2 = "dnabert-2" in model_name_or_path.lower()

    if is_nt:
        model = transformers.AutoModelForMaskedLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
    elif is_dnabert2:
        config = transformers.models.bert.configuration_bert.BertConfig.from_pretrained(
            model_name_or_path
        )
        model = transformers.AutoModelForMaskedLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            config=config,
        )
    else:
        model = transformers.AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )

    n_gpu = torch.cuda.device_count()
    if n_gpu >= 1:
        model = nn.DataParallel(model)
        model.to("cuda")
        n_cpu = 0
    else:
        model.to("cpu")
        n_cpu = 1

    train_loader = util_data.DataLoader(
        dna_sequences,
        batch_size=batch_size * (n_gpu + n_cpu),
        shuffle=False,
        num_workers=2 * (n_gpu + n_cpu),
    )
    for j, batch in enumerate(tqdm.tqdm(train_loader)):
        with torch.no_grad():
            token_feat = tokenizer.batch_encode_plus(
                batch,
                max_length=model_max_length,
                return_tensors="pt",
                padding="longest",
                truncation=True,
            )
            input_ids = token_feat["input_ids"]
            if not is_hyenadna:
                attention_mask = token_feat["attention_mask"]
            if n_gpu:
                input_ids = input_ids.cuda()
                if not is_hyenadna:
                    attention_mask = attention_mask.cuda()

            if is_hyenadna:
                model_output = model.forward(input_ids=input_ids)[0].detach().cpu()
                attention_mask = torch.ones(
                    size=(model_output.shape[0], model_output.shape[1], 1), device="cpu"
                )
            else:
                model_output = (
                    model.forward(input_ids=input_ids, attention_mask=attention_mask)[0]
                    .detach()
                    .cpu()
                )
                attention_mask = attention_mask.unsqueeze(-1).detach().cpu()

            embedding = torch.sum(model_output * attention_mask, dim=1) / torch.sum(
                attention_mask, dim=1
            )

            if j == 0:
                embeddings = embedding
            else:
                embeddings = torch.cat((embeddings, embedding), dim=0)

    embeddings = np.array(embeddings.detach().cpu())

    # reorder the embeddings
    embeddings = embeddings[np.argsort(idx)]

    return embeddings


def align_labels_via_hungarian_algorithm(true_labels, predicted_labels):
    """
    Aligns the predicted labels with the true labels using the Hungarian algorithm.

    Args:
    true_labels (list or array): The true labels of the data.
    predicted_labels (list or array): The labels predicted by a clustering algorithm.

    Returns:
    dict: A dictionary mapping the predicted labels to the aligned true labels.
    """
    true_labels, predicted_labels = (
        np.array(true_labels, dtype=int),
        np.array(predicted_labels, dtype=int),
    )

    # Create a confusion matrix
    max_label = max(max(true_labels), max(predicted_labels)) + 1
    confusion_matrix = np.zeros((max_label, max_label), dtype=int)

    for true_label, predicted_label in zip(true_labels, predicted_labels):
        confusion_matrix[true_label, predicted_label] += 1

    # Apply the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(confusion_matrix, maximize=True)

    # Create a mapping from predicted labels to true labels
    label_mapping = {
        predicted_label: true_label
        for true_label, predicted_label in zip(row_ind, col_ind)
    }

    return label_mapping


def compute_similarity_matrix(features1, features2=None, metric="dot", chunk_size=512):
    """
    Compute the similarity matrix between two sets of features
    """
    if features2 is None:
        features2 = features1

    if metric == "dot":
        if isinstance(features1, tuple):
            features1, features2 = features1[0], features2[0]

        if chunk_size:
            similarities = np.zeros((features1.shape[0], features2.shape[0]))
            for i in range(0, features1.shape[0], chunk_size):
                similarities[i : i + chunk_size, :] = (
                    features1[i : i + chunk_size, :] @ features2.T
                )
        else:
            similarities = np.dot(features1, features2.T)

    elif metric == "l1":
        if isinstance(features1, tuple):
            features1, features2 = features1[0], features2[0]

        similarities = np.exp(-distance.cdist(features1, features2, "minkowski", p=1))

    elif metric == "l2":
        if isinstance(features1, tuple):
            features1, features2 = features1[0], features2[0]

        similarities = np.exp(-distance.cdist(features1, features2, "minkowski", p=2.0))

    elif metric == "squared_l2":
        if isinstance(features1, tuple):
            features1, features2 = features1[0], features2[0]

        similarities = np.exp(
            -(distance.cdist(features1, features2, "minkowski", p=2.0) ** 2)
        )

    elif metric == "mahalanobis":
        if chunk_size:
            similarities = np.zeros((features1[0].shape[0], features2[0].shape[0]))
            for i in range(0, features1[0].shape[0], chunk_size):
                left_mean = features1[0][i : i + chunk_size, :][:, np.newaxis, :]
                left_cov = features1[1][i : i + chunk_size, :][:, np.newaxis, :]
                right_mean = features2[0][np.newaxis, :, :]
                right_cov = features2[1][np.newaxis, :, :]

                mean_squared_diff = (left_mean - right_mean) ** 2 * (
                    0.5 / (left_cov + right_cov)
                )
                log_expectation = -0.5 * (mean_squared_diff).sum(axis=-1)
                similarities[i : i + chunk_size, :] = np.exp(log_expectation)

        else:
            mean_squared_diff = (features1[0] - features2[0]) ** 2 * (
                0.5 / (features1[1] + features2[2])
            )
            log_expectation = -0.5 * (mean_squared_diff).sum(axis=-1)
            similarities = np.exp(log_expectation)

    else:
        raise ValueError("Invalid metric!")

    return similarities

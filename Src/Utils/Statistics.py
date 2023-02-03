import torch
import numpy as np
import random

# helper functions to compute statistics

# calculate the confusion matrix
def confusion_matrix(accuracy, labels):

    TN = np.sum((labels == 0) & (accuracy == True))
    FN = np.sum((labels == 1) & (accuracy == False))
    TP = np.sum((labels == 1) & (accuracy == True))
    FP = np.sum((labels == 0) & (accuracy == False))

    return {'TN' : TN, 'FN' : FN, 'TP' : TP, 'FP' : FP}

# convert to rates
def accuracy_rates(TN, FN, TP, FP):

    FAR = FP / float(FP + TN) if FP + TN else 0.0
    TPR = 1 - FAR
    FRR = FN/ float(FN + TP) if FN + TP else 0.0
    TNR = 1 - FRR

    ACER = np.mean([FAR, FRR])

    return {'TNR' : TNR, 'FRR' : FRR, 'FAR' : FAR, 'TPR' : TPR, 'ACER' : ACER}

# direct calculation for simplicity
def confusion_and_accuracy(accuracy, labels):

    TN = np.sum((labels == 0) & (accuracy == True))
    FN = np.sum((labels == 1) & (accuracy == False))
    TP = np.sum((labels == 1) & (accuracy == True))
    FP = np.sum((labels == 0) & (accuracy == False))

    FAR = FP / float(FP + TN) if FP + TN else 0.0
    TPR = 1 - FAR
    FRR = FN/ float(FN + TP) if FN + TP else 0.0
    TNR = 1 - FRR

    ACER = np.mean([FAR, FRR])

    return {'TNR' : TNR, 'FRR' : FRR, 'FAR' : FAR, 'TPR' : TPR, 'ACER' : ACER}

# helper function to extract the valid items
# necessary when some labels are missing 
def sample_valid_labels(embeddings, labels, mask = -1):

    # get the valid indeces
    valid_indeces = (labels != mask).nonzero()

    # apply the mask sampling
    embeddings = torch.cat([embeddings[i] for i in valid_indeces])
    labels = torch.cat([labels[i] for i in valid_indeces])

    return embeddings, labels

# helper file to extract the epoch end outputs
def epoch_end_extract(eval):

    data = {}
    for key in eval[0].keys():
        # print(key, [x[key] for x in eval])
        data[key] = [x[key] for x in eval]

    return data

# function to generate a list of randomized samples
# utility is torch.nonzero is really slow
def randomized_pairwise_sampler(labels, samples):

    same_ID = []
    diff_ID = []

    print('Generating indeces')
    indeces = range(0, len(labels))

    valid_same = True
    valid_diff = True

    # go through the shuffled indeces
    while valid_same and valid_diff:

        idx = random.sample(indeces, 1)

        if labels[idx] == False and valid_diff:
            diff_ID.append(idx)
            valid_diff == len(diff_ID) < int(samples)/2

            if not valid_diff:
                print(idx, labels[idx], valid_same, valid_diff)

        elif labels[idx] == True and valid_same:
            same_ID.append(idx)
            valid_same == len(same_ID) < int(samples)/2

            if not valid_same:
                print(idx, labels[idx], valid_same, valid_diff)

        elif not (valid_diff and valid_same):
            break

    return same_ID, diff_ID

# helper function to do pairwise comparison
def pairwise_distance(embeddings, labels, sample = 100000):

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    similarities = []
    targets = []
    embed_comp_1 = []
    embed_comp_2 = []
    labels_1 = []
    labels_2 = []

    # generate a ledger of comparison indeces through a loop
    num_eval_embeddings = len(embeddings)
    # for idx, embedding in enumerate(embeddings[:-1]):
    for idx in range(num_eval_embeddings-1):

        # repeat the primary embedding and fetch the comparison ones
        embed_comp_1 += [idx] * (len(embeddings) - idx -1)
        embed_comp_2 += list(range(idx+1, num_eval_embeddings))

        # repeat the primary label, fetch the comparison ones and determine if they are same
        labels_1.append(labels[idx].repeat(len(embeddings) -idx -1))
        labels_2.append(labels[idx+1:])

    # stack the label tensors
    labels_1 = torch.cat(labels_1)
    labels_2 = torch.cat(labels_2)
    label_comp = labels_1 == labels_2

    # if we have more objects than sample size, we need to randomize the comparisons
    if sample != None and sample < len(label_comp):
        # find same vs diff pairs
        # note nonzero is slow, potential workaround is to count up instead
        # this shifts number of label comparisons to be closer to sample size (for well distributed data)
        same_ID = label_comp.nonzero()
        diff_ID = (label_comp == False).nonzero()

        # to ensure proper metrics, sample for FRR and FAR seperately
        num_same = len(same_ID)
        num_diff = len(diff_ID)
        same_indeces = list(range(0, num_same))
        diff_indeces = list(range(0, num_diff))

        # use the random sample function to effciently generate the randomized pairs
        same_sample = random.sample(same_indeces, same_sample)
        diff_sample = random.sample(diff_indeces, diff_sample)

        # convert back to indeces
        same_indeces = [same_ID[idx] for idx in same_sample]
        diff_indeces = [diff_ID[idx] for idx in diff_sample]
        indeces = same_indeces + diff_indeces

        # fetch only the correct indeces
        embed_comp_1 = [embed_comp_1[i] for i in indeces]
        embed_comp_2 = [embed_comp_2[i] for i in indeces]
        label_comp = [label_comp[i] for i in indeces]
        label_comp = torch.stack(label_comp)

        # fetch only the correct indeces
        embed_comp_1 = [embed_comp_1[i] for i in indeces]
        embed_comp_2 = [embed_comp_2[i] for i in indeces]
        label_comp = [label_comp[i] for i in indeces]
        label_comp = torch.stack(label_comp)

    # convert the list back to tensors
    for k in range(len(embed_comp_1)):
        embed_comp_1[k] = embeddings[embed_comp_1[k]]
        embed_comp_2[k] = embeddings[embed_comp_2[k]]

    # stack the embedding tensors
    embed_comp_1 = torch.stack(embed_comp_1)
    embed_comp_2 = torch.stack(embed_comp_2)

    # calculate similarities in batch
    similarities = cos(embed_comp_1, embed_comp_2)

    return similarities, label_comp


# helper function to pre-compute the similarity ledgers
def generate_pairwise_ledger(labels):

    embed_comp_1 = []
    embed_comp_2 = []
    labels_1 = []
    labels_2 = []

    # generate a ledger of comparison indeces through a loop
    num_eval = len(labels)
    # for idx, embedding in enumerate(embeddings[:-1]):
    for idx in range(num_eval-1):

        # repeat the primary embedding and fetch the comparison ones
        embed_comp_1 += [idx] * (num_eval - idx -1)
        embed_comp_2 += list(range(idx+1, num_eval))

        # repeat the primary label, fetch the comparison ones and determine if they are same
        labels_1.append(labels[idx].repeat(num_eval -idx -1))
        labels_2.append(labels[idx+1:])

    # stack the label tensors
    labels_1 = torch.cat(labels_1)
    labels_2 = torch.cat(labels_2)
    label_comp = labels_1 == labels_2

    # find same vs diff pairs
    # note nonzero is slow, potential workaround is to count up instead
    # this shifts number of label comparisons to be closer to sample size (for well distributed data)
    same_ID = label_comp.nonzero()
    diff_ID = (label_comp == False).nonzero()

    ledger = {'embed_comp_1': embed_comp_1, 'embed_comp_2' : embed_comp_2, 'label_comp' : label_comp, 'same_ID': same_ID, 'diff_ID' : diff_ID}

    return ledger

# helper function to do pairwise comparison
def pairwise_distance_ledger(embeddings, labels, ledger, samples = 100000):

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    similarities = []
    targets = []
    # embed_comp_1 = ledger['embed_comp_1']
    # embed_comp_2 = ledger['embed_comp_2']
    # label_comp = ledger['label_comp']
    embed_comp_1 = []
    embed_comp_2 = []
    label_comp = []
    same_ID = ledger['same_ID']
    diff_ID = ledger['diff_ID']

    # to ensure proper metrics, sample for FRR and FAR seperately
    num_same = len(same_ID)
    num_diff = len(diff_ID)
    same_indeces = range(0, num_same)
    diff_indeces = range(0, num_diff)

    # determine number of datapoints based off ideal split and max available
    # have to put an off by 1 condition in case batch is ID identical or 100% sparse
    ceiling = np.max([1, np.min([num_same, num_diff])])
    same_sample = np.min([num_same, ceiling, int(samples/2)])
    diff_sample = np.min([num_diff, ceiling, int(samples/2)])

    # use the random sample function to effciently generate the randomized pairs
    same_sample = random.sample(same_indeces, same_sample)
    diff_sample = random.sample(diff_indeces, diff_sample)

    # convert back to indeces
    same_indeces = [same_ID[idx] for idx in same_sample]
    diff_indeces = [diff_ID[idx] for idx in diff_sample]
    indeces = same_indeces + diff_indeces

    # fetch only the correct indeces
    # note we need to map back to embeddings from the ledger
    for i in indeces:
        embed_comp_1.append(embeddings[ledger['embed_comp_1'][i]])
        embed_comp_2.append(embeddings[ledger['embed_comp_2'][i]])
        label_comp.append(ledger['label_comp'][i])
    #print('mapped')

    # stack the embedding tensors
    embed_comp_1 = torch.stack(embed_comp_1)
    embed_comp_2 = torch.stack(embed_comp_2)
    label_comp = torch.stack(label_comp)


    # calculate similarities in batch
    similarities = cos(embed_comp_1, embed_comp_2)
    label_comp = label_comp[:,0]

    return similarities, label_comp

# helper function to estimate best accuracy on pairwise comparison
def pairwise_accuracy(similarities, labels, th_steps = 50):

    best_acc = 0
    best_th = 0

    # go through the thresholds
    thresholds = np.linspace(.125,.975,th_steps)

    for th in thresholds:
        # determine pass/fail based upon similarity
        y_test = (similarities >= th)
        acc = np.mean((y_test == labels).astype(int))

        # keep track of the best scores
        if acc > best_acc:
            best_acc = acc
            best_th = th
            rates = confusion_and_accuracy(acc, labels)

    output = {'accuracy' : best_acc, 'threshold': best_th}
    return output

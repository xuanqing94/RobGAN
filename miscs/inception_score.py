# copied from https://github.com/sbarratt/inception-score-pytorch
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

import numpy as np
from scipy.stats import entropy

def inception_score(model, imgs, batch_size=200, resize=True, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    n, c, h, w = imgs.shape

    assert batch_size > 0
    assert n > batch_size
    assert n % batch_size == 0

    dtype = torch.cuda.FloatTensor

    # Load inception model
    #inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    #inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((n, 1000))

    for i in range(n // batch_size):
        batch = torch.from_numpy(imgs[i*batch_size:(i+1)*batch_size])
        batch = batch.cuda()
        batchv = Variable(batch)
        preds[i*batch_size:(i+1)*batch_size] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (n // splits): (k+1) * (n // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


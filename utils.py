# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# compute the current classification accuracy
def compute_acc(preds, labels):
    correct = 0
    preds_ = preds.data.max(1)[1]
    correct = preds_.eq(labels.data).cpu().sum()
    acc = float(correct) / float(len(labels.data)) * 100.0
    return acc


def dice_loss(input, target):
    input[input < 0.5] = 0
    input[input > 0.5] = 1
    scores  = []
    smooth = 1.
    for idx in range(0, input.shape[0]):
        for _iter in range(0, target.shape[0]):

            iflat = input[idx,0,:,:].view(-1)
            tflat = target[_iter,0,:,:].view(-1)
            intersection = (iflat * tflat).sum()

            scores.append(1 - ((2. * intersection + smooth) /
                        (iflat.sum() + tflat.sum() + smooth)))
    return sum(scores)/len(scores)
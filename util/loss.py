from torch.nn import functional as F 

def cos_loss(m1, m2):
    return F.cosine_similarity(m1, m2, dim=1)

def euc_loss(m1, m2):
    return F.pairwise_distance(m1, m2, p=2)
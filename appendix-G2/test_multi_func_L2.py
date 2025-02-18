import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import math
from matplotlib import pyplot as plt
import numpy as np
import random

def prime(end):
    prime_list = []
    for i in range(end):
        if i == 0 or i == 1:
            continue
        else:
            for j in range(2, int(i/2)+1):
                if i % j == 0:
                    break
            else:
                prime_list.append(i)
    return prime_list


train_len = 50
total_len = 200
d = 32


def get_label(size: torch.Size):
    batch_size, seq_len, _ = size
    indices = torch.arange(seq_len).long()
    target = torch.zeros(seq_len, seq_len)

    "=============================================="

    # testing function: attn = 1 if j+c == i else 0
    # c = int(sys.argv[1])
    # assert c >= 0
    # labels = ((indices.unsqueeze(1) - indices.unsqueeze(0)) == c)
    # labels = (labels.float() * 2 * math.log(train_len))   # 0 or 2logN
    # target += labels

    # testing function: attn = 1 if j > i-c else 0
    # c = int(sys.argv[1])
    # assert c >= 2
    # labels = ((indices.unsqueeze(1) - indices.unsqueeze(0)) < c)
    # labels = (labels.float() * 2 * math.log(train_len))   # 0 or 2logN
    # target += labels

    # testing function: attn = 1 if (i-j) % c1 == c2 else 0   # nice when c is small
    c1 = int(sys.argv[1])
    c2 = int(sys.argv[2])
    assert c1 > 0 and c2 >= 0
    labels = (((indices.unsqueeze(1) - indices.unsqueeze(0)) % c1) == c2)
    labels = (labels.float() * 2 * math.log(train_len))   # 0 or 2logN
    target += labels

    # ==============

    # testing function: attn = 1 if j < i-c else 0
    # c = int(sys.argv[1])
    # assert c >= 0
    # labels = ((indices.unsqueeze(1) - indices.unsqueeze(0)) > c)
    # labels = (labels.float() * 2 * math.log(train_len))   # 0 or 2logN
    # target += labels

    # testing function: attn = 1 if i-j in numbers series else 0
    # numbers = torch.tensor(prime(seq_len), dtype=torch.long)
    # labels = ((indices.unsqueeze(1) - indices.unsqueeze(0)).unsqueeze(0) == numbers.view(-1, 1, 1)).sum(dim=0) > 0
    # labels = (labels.float() * 2 * math.log(train_len))   # 0 or 2logN
    # target += labels

    "=============================================="

    target = target.unsqueeze(0).expand(size)
    mask = ((indices.unsqueeze(1) - indices.unsqueeze(0)) >= 0)
    mask = mask.unsqueeze(0).expand(size)

    return target, mask

torch.set_printoptions(sci_mode=False)
torch.manual_seed(0)


class pTAp(nn.Module):
    def __init__(self, d, total_len, rank=None) -> None:
        super().__init__()
        self.embedding = nn.Parameter( (torch.rand(total_len, d) * 2 - 1))

        if rank is None:
            rank = d

        self.W_Q = nn.Parameter( (torch.rand(d, rank) * 2 - 1) / math.sqrt(rank) )
        self.W_K = nn.Parameter( (torch.rand(d, rank) * 2 - 1) / math.sqrt(rank) )
        
    def forward(self, q_indices: torch.LongTensor, k_indices: torch.LongTensor):
        # q_indices: batch_size, window_len
        # k_indices: batch_size, window_len

        q = torch.matmul(self.embedding[q_indices], self.W_Q)
        k = torch.matmul(self.embedding[k_indices], self.W_K)
        attn_score = torch.matmul(q, k.transpose(1, 2))

        mask_value = torch.full([], -float("inf"))   # torch.finfo(attn_score.dtype).min, dtype=attn_score.dtype
        attn_score = torch.where(torch.ones(attn_score.size()[1:]).tril().unsqueeze(0).bool(), attn_score, mask_value)

        return attn_score
    

product = pTAp(d, total_len)
optimizer = torch.optim.Adam(product.parameters(), lr=1e-3)

test_len_groups = [50, 100, 150]

loss_func = nn.MSELoss()

batch_size = 64
for step in range(15_000):
    window_s = torch.randint(0, total_len-train_len+1, (batch_size, 1))  # position offset 
    indices = window_s + torch.arange(train_len).long().unsqueeze(0)

    attn_logits = product(indices, indices)
    labels, mask = get_label(attn_logits.size())
    loss = loss_func(attn_logits[mask], labels[mask])

    loss = loss + 0.01 * product.embedding.pow(2).mean()
    loss = loss + 0.01 * product.W_Q.pow(2).mean() + 0.01 * product.W_K.pow(2).mean()


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (step+1) % 1000 == 0:
        with torch.no_grad():
            print("==="*10, "step", step+1, "==="*10)
            for test_l in test_len_groups:
                window_s = torch.arange(total_len-test_l+1).long().unsqueeze(1)   # all possible idx
                indices = window_s + torch.arange(test_l).long().unsqueeze(0)

                attn_logits = product(indices, indices)
                labels, mask = get_label(attn_logits.size())


                test_loss = loss_func(attn_logits[mask], labels[mask])

                print(f"{test_loss.item():.10f}", end=", ")

            print()
            
# show attn
with torch.no_grad():
    for i, test_l in enumerate(test_len_groups):
        window_s = torch.arange(total_len-test_l+1).long().unsqueeze(1)   # all possible idx
        indices = window_s + torch.arange(test_l).long().unsqueeze(0)

        attn_logits = product(indices, indices)
        # print(attn_logits[0])
        # attn_weights /= attn_weights.max(dim=1, keepdim=True)[0]

        fig, ax = plt.subplots()
        fig.set_size_inches(20, 20)
        fig.set_dpi(150)
        im = ax.imshow(attn_logits[0].numpy())
        cb = fig.colorbar(im)
        ax.tick_params(axis="both", labelsize=25) 
        cb.ax.tick_params(labelsize=40) 
  
        fig.savefig(f"./figures/test{test_l}.png")

        fig, ax = plt.subplots()
        fig.set_size_inches(20, 20)
        fig.set_dpi(150)
        im = ax.imshow(attn_logits.mean(dim=0).numpy())
        cb = fig.colorbar(im)
        ax.tick_params(axis="both", labelsize=25) 
        cb.ax.tick_params(labelsize=40) 
  
        fig.savefig(f"./figures/test{test_l}_mean.png")


    # p_norm = torch.linalg.vector_norm(product.embedding, dim=-1).numpy()
    # plt.plot(np.arange(p_norm.shape[0]), p_norm)
    # plt.show()
   
results_d32 = {"j=i-c": {"c=1": [0.0124985492, 496.2664794922, 755.2360839844,], 
                         "c=5": [0.0271074250, 333.3888244629, 606.5682983398,], 
                         "c=25": [0.0000022057, 0.7152619362, 1.0706132650,]},
                "j>i-c": {"c=2": [0.0083255535, 303.7139282227, 414.6399841309,],
                          "c=5": [0.0255450439, 216.1955566406, 381.6783447266,],
                          "c=25": [0.0078058084, 9.3571901321, 15.1211357117,]},
                "(i-j)=c_2 mod c_1": {"c_1=3, c_2=0": [0.0000034845, 0.0010645119, 0.0053957077,],
                                      "c_1=5, c_2=0": [0.0000032848, 0.0002877587, 0.0015938719,],
                                      "c_1=3, c_2=2": [0.0000013159, 0.0000026057, 0.0000058817,],
                                      "c_1=5, c_2=2": [0.0000028226, 0.0000027044, 0.0000029815,],},
                "j<i-c": {"c=0": [0.0708710179, 229.7775573730, 368.5939636230,],
                          "c=5": [0.0573513247, 95.5051727295, 212.6814727783,],
                          "c=25": [0.0000146985, 9.1086015701, 13.8165292740,]},
                "i-j is prime": {" ": [0.0521395244, 7.8534722328, 17.9720077515,],},
                "combined": {"j=i-1 | (i-j)=0 mod 3": [0.0234131571, 234.0101318359, 420.2164001465,],
                             "j=i-10 | (i-j)=0 mod 3": [0.1561257094, 4.5674924850, 17.3307113647,],
                             "j=i-1 | (i-j)=0 mod 5": [0.0284165759, 160.3611602783, 290.1488037109,],
                             "j=i-12 | (i-j)=0 mod 5": [0.1483774185, 2.0993382931, 5.5044059753,],}
}

results_d256 = {"j=i-c": {"c=1": [0.0000045789, 0.2099193931, 0.3275869489,], 
                         "c=5": [0.0000039742, 0.2041597813, 0.3235522509,], 
                         "c=25": [0.0000049315, 0.1646348983, 0.2828634381,]},
                "j>i-c": {"c=2": [0.0000372034, 0.3495599926, 0.5547776222, ],
                          "c=5": [0.0001623197, 0.6075839400, 1.0019609928, ],
                          "c=25": [0.0000049899, 1.8029826880, 3.1008999348, ]},
                "(i-j)=c_2 mod c_1": {"c_1=3, c_2=0": [0.0003711144, 2.8137290478, 6.8732824326,],
                                      "c_1=5, c_2=0": [0.0003158526, 2.3210136890, 5.1842579842,],
                                      "c_1=3, c_2=2": [0.0006015065, 3.1011548042, 7.2998561859,],
                                      "c_1=5, c_2=2": [0.0000395538, 1.9786781073, 4.6821579933,],},
                "j<i-c": {"c=0": [0.0003972818, 5.5996322632, 16.4334640503,],
                          "c=5": [0.0002407077, 9.2956085205, 21.4885139465,],
                          "c=25": [0.0000289284, 14.2473096848, 26.0415172577,]},
                "i-j is prime": {" ": [0.0000668037, 3.6927046776, 7.2145252228,],},
                "combined": {"j=i-1 | (i-j)=0 mod 3": [0.0000087227, 3.2345695496, 7.5253567696, ],
                             "j=i-10 | (i-j)=0 mod 3": [0.0002122433, 3.2348296642, 7.5272789001, ],
                             "j=i-1 | (i-j)=0 mod 5": [0.0000581382, 2.7456986904, 5.7526345253,],
                             "j=i-12 | (i-j)=0 mod 5": [0.0000177166, 2.7219662666, 5.7598857880, ],}
}



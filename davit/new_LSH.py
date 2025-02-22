import torch
import torch.nn as nn
from collections import deque
from tools import common
import torch.nn.functional as F
class NonLocalSparseAttention(nn.Module):
    def __init__(self, n_hashes=4, channels=12, k_size=3, reduction=4, chunk_size=144, conv=common.default_conv,
                 res_scale=1):
        super(NonLocalSparseAttention, self).__init__()
        self.chunk_size = chunk_size  # 每个chunk有144个元素的hash值
        self.n_hashes = n_hashes  # hash值为4维
        self.reduction = reduction
        self.res_scale = res_scale
        self.conv_match = common.BasicBlock(conv, channels, channels // reduction, k_size, bn=False, act=None)
        self.conv_assembly = common.BasicBlock(conv, channels, channels, 1, bn=False, act=None)
        self.conv1 = nn.Conv2d(32, 8, 3, 1, 1, bias=True)
        # 作为对比 标准Non-local如下
        # self.conv_match1 = common.BasicBlock(conv, channel, channel//reduction, 1, bn=False, act=nn.PReLU())
        # self.conv_match2 = common.BasicBlock(conv, channel, channel//reduction, 1, bn=False, act = nn.PReLU())
        # self.conv_assembly = common.BasicBlock(conv, channel, channel, 1,bn=False, act=nn.PReLU())

    def LSH(self, hash_buckets, x):
        # x: [N,H*W,C]
        N = x.shape[0]  # batch size
        device = x.device

        # generate random rotation matrix
        rotations_shape = (1, x.shape[-1], self.n_hashes, hash_buckets // 2)  # [1,C,n_hashes,hash_buckets//2]
        random_rotations = torch.randn(rotations_shape, dtype=x.dtype, device=device).expand(N, -1, -1,
                                                                                             -1)  # [N, C, n_hashes, hash_buckets//2]

        # locality sensitive hashing [n hw c]*[N, C, n_hashes, hash_buckets//2]
        rotated_vecs = torch.einsum('btf,bfhi->bhti', x,
                                    random_rotations)  # [N, n_hashes, H*W, hash_buckets//2]，把channel维度融掉了（hw乘以其对应的数进行旋转），对应于论文流程图中的求和步骤
        rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)  # [N, n_hashes, H*W, hash_buckets]
        # 为什么要这样做呢（又有正，又有负）？可以参考
        # [42] Kengo Terasawa and Yuzuru Tanaka. Spherical lsh for approximate nearest neighbor search on unit hypersphere. In Workshop on Algorithms and Data Structures, pages 27–38. Springer, 2007. 3
        # 的附录，对应于orthoplex情景

        # get hash codes
        hash_codes = torch.argmax(rotated_vecs,
                                  dim=-1)  # [N, n_hashes, H*W, hash_buckets]->[N,n_hashes,H*W]求得每个hash bucket中最大的值的位置 作为该feature map像素点的hash值

        # add offsets to avoid hash codes overlapping between hash rounds 加了一点偏移量，防止hash code重叠
        offsets = torch.arange(self.n_hashes, device=device)  # 生成【0，1，2，3】数组
        offsets = torch.reshape(offsets * hash_buckets, (1, -1, 1))  # 【0，1*hb,3*hb,3*hb】  形状是（1，4，1）
        hash_codes = torch.reshape(hash_codes + offsets, (N, -1,))  # [N,n_hashes(这个维度和offsets一样),H*W]->[N,n_hashes*H*W]

        return hash_codes

    def add_adjacent_buckets(self, x):
        # 这个函数用于把相邻的bucket相连
        x_extra_back = torch.cat([x[:, :, -1:, ...], x[:, :, :-1, ...]], dim=2)  # 把倒数第一行移到了第一行的位置 相当于向下移动一行
        x_extra_forward = torch.cat([x[:, :, 1:, ...], x[:, :, :1, ...]], dim=2)  # 把第一行移到了倒数第一行的位置 相当于向上移动一行
        return torch.cat([x, x_extra_back, x_extra_forward], dim=3)  # 将这三个东西沿着行的方向进行拼接
        # 这个操作十分巧妙地将第i组 第i-1和i+1组放在了一行里面 拼接了这三个组

    def forward(self, input):
        N, _, H, W = input.shape
        x_embed = self.conv_match(input).view(N, -1, H * W).contiguous().permute(0, 2, 1)  # channel数 ➗4了 [N,h*w,c/4]
        y_embed = self.conv_assembly(input).view(N, -1, H * W).contiguous().permute(0, 2, 1)  # channel数没有变 [N,h*w,c]
        # contiguous：view只能作用在contiguous的variable上，如果在view之前调用了transpose、permute等，就需要调用contiguous()来返回一个contiguous copy；
        # 这儿为什么不是在permute之后采用contigious呢？不是很懂
        L, C = x_embed.shape[-2:]  # L是H*W，且C是channel/4

        # number of hash buckets/hash bits 计算有多少个桶呢 最多128个
        hash_buckets = min(L // self.chunk_size + (L // self.chunk_size) % 2, 128)  # 保障hash_buckets（bucket的数量）是偶数

        # get assigned hash codes/bucket number
        hash_codes = self.LSH(hash_buckets, x_embed)  # [N,n_hashes*H*W]
        hash_codes = hash_codes.detach()  # 计算过程不需要反向传播

        # group elements with same hash code by sorting
        _, indices = hash_codes.sort(dim=-1)  # [N,n_hashes*H*W] sort以升序排列，返回值为value-tensor和indice-tensor
        _, undo_sort = indices.sort(dim=-1)  # undo_sort to recover original order
        # 这里返回的是 【N,n_hashes*H*W】这一次返回值是原来的hash_codes中每一个值它的大小在整个序列里面的排名，如果给了这个序列按顺序排列的结果，那可以根据这个undo-sort列表，还原出原始的序列来。
        mod_indices = (indices % L)  # now range from (0->H*W)
        x_embed_sorted = common.batched_index_select(x_embed, mod_indices)  # [N,n_hashes*H*W,C]
        y_embed_sorted = common.batched_index_select(y_embed, mod_indices)  # [N,n_hashes*H*W,C*4]
        # def batched_index_select(values, indices):
        #     last_dim = values.shape[-1]
        #     return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))
        # None的作用是在最后增加一维，类似于np.newaxis

        # pad the embedding if it cannot be divided by chunk_size
        padding = self.chunk_size - L % self.chunk_size if L % self.chunk_size != 0 else 0
        x_att_buckets = torch.reshape(x_embed_sorted, (N, self.n_hashes, -1, C))  # [N, n_hashes, H*W,C]
        y_att_buckets = torch.reshape(y_embed_sorted, (N, self.n_hashes, -1, C * self.reduction))
        if padding:
            pad_x = x_att_buckets[:, :, -padding:, :].clone()
            pad_y = y_att_buckets[:, :, -padding:, :].clone()
            x_att_buckets = torch.cat([x_att_buckets, pad_x], dim=2)
            y_att_buckets = torch.cat([y_att_buckets, pad_y], dim=2)  # 把最后几个作为pad来补足

        x_att_buckets = torch.reshape(x_att_buckets, (
        N, self.n_hashes, -1, self.chunk_size, C))  # [N, n_hashes, num_chunks, chunk_size, C]
        y_att_buckets = torch.reshape(y_att_buckets, (N, self.n_hashes, -1, self.chunk_size, C * self.reduction))

        x_match = F.normalize(x_att_buckets, p=2, dim=-1, eps=5e-5)  # L2归一化
        # [N, n_hashes, num_chunks, chunk_size, C]

        # allow attend to adjacent buckets
        # 论文中We then apply the Non-Local (NL) operation within the bucket that the query pixel belongs to, or across adjacent buckets after sorting.
        # 为了可以搜索相邻的组
        x_match = self.add_adjacent_buckets(x_match)
        y_att_buckets = self.add_adjacent_buckets(y_att_buckets)

        # unormalized attention score
        raw_score = torch.einsum('bhkie,bhkje->bhkij', x_att_buckets,
                                 x_match)  # [N, n_hashes, num_chunks, chunk_size, chunk_size*3]

        # softmax
        bucket_score = torch.logsumexp(raw_score, dim=-1, keepdim=True)  # logsumexp实际上是针对max函数的一种平滑操作
        score = torch.exp(raw_score - bucket_score)  # (after softmax)
        bucket_score = torch.reshape(bucket_score, [N, self.n_hashes, -1])

        # attention
        ret = torch.einsum('bukij,bukje->bukie', score, y_att_buckets)  # [N, n_hashes, num_chunks, chunk_size, C]
        ret = torch.reshape(ret, (N, self.n_hashes, -1, C * self.reduction))

        # if padded, then remove extra elements
        if padding:
            ret = ret[:, :, :-padding, :].clone()
            bucket_score = bucket_score[:, :, :-padding].clone()

        # recover the original order
        ret = torch.reshape(ret, (N, -1, C * self.reduction))  # [N, n_hashes*H*W,C]
        bucket_score = torch.reshape(bucket_score, (N, -1,))  # [N,n_hashes*H*W]
        ret = common.batched_index_select(ret, undo_sort)  # [N, n_hashes*H*W,C]
        bucket_score = bucket_score.gather(1, undo_sort)  # [N,n_hashes*H*W]

        # weighted sum multi-round attention
        ret = torch.reshape(ret, (N, self.n_hashes, L, C * self.reduction))  # [N, n_hashes*H*W,C]
        bucket_score = torch.reshape(bucket_score, (N, self.n_hashes, L, 1))
        probs = nn.functional.softmax(bucket_score, dim=1)
        ret = torch.sum(ret * probs, dim=1)

        ret = ret.permute(0, 2, 1).view(N, -1, H, W).contiguous() * self.res_scale + input
        return ret
if __name__ == '__main__':
    input = torch.randn(64,24,96,96)
    model = NonLocalSparseAttention()
    print(model)
    out = model(input)
    print(out.shape)

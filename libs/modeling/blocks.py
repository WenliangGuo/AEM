import math
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from .weight_init import trunc_normal_


class MaskedConv1D(nn.Module):
    """
    Masked 1D convolution. Interface remains the same as Conv1d.
    Only support a sub set of 1d convs
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros'
    ):
        super().__init__()
        # element must be aligned
        assert (kernel_size % 2 == 1) and (kernel_size // 2 == padding)
        # stride
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode)
        # zero out the bias term if it exists
        if bias:
            torch.nn.init.constant_(self.conv.bias, 0.)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()
        # input length must be divisible by stride
        assert T % self.stride == 0

        # conv
        out_conv = self.conv(x)
        # compute the mask
        if self.stride > 1:
            # downsample the mask using nearest neighbor
            out_mask = F.interpolate(
                mask.to(x.dtype), size=out_conv.size(-1), mode='nearest'
            )
        else:
            # masking out the features
            out_mask = mask.to(x.dtype)

        # masking the output, stop grad to mask
        out_conv = out_conv * out_mask.detach()
        out_mask = out_mask.bool()
        return out_conv, out_mask


class LayerNorm(nn.Module):
    """
    LayerNorm that supports inputs of size B, C, T
    """
    def __init__(
        self,
        num_channels,
        eps = 1e-5,
        affine = True,
        device = None,
        dtype = None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(
                torch.ones([1, num_channels, 1], **factory_kwargs))
            self.bias = nn.Parameter(
                torch.zeros([1, num_channels, 1], **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        assert x.dim() == 3
        assert x.shape[1] == self.num_channels

        # normalization along C channels
        mu = torch.mean(x, dim=1, keepdim=True)
        res_x = x - mu
        sigma = torch.mean(res_x**2, dim=1, keepdim=True)
        out = res_x / torch.sqrt(sigma + self.eps)

        # apply weight and bias
        if self.affine:
            out *= self.weight
            out += self.bias

        return out


# helper functions for Transformer blocks
def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    # return a tensor of size 1 C T
    return torch.FloatTensor(sinusoid_table).unsqueeze(0).transpose(1, 2)


# attention / transformers
class MaskedMHA(nn.Module):
    """
    Multi Head Attention with mask

    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(
        self,
        n_embd,          # dimension of the input embedding
        n_head,          # number of heads in multi-head self-attention
        attn_pdrop=0.0,  # dropout rate for the attention map
        proj_pdrop=0.0   # dropout rate for projection op
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)

        # key, query, value projections for all heads
        # it is OK to ignore masking, as the mask will be attached on the attention
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # calculate query, key, values for all heads in batch
        # (B, nh * hs, T)
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # move head forward to be the batch dim
        # (B, nh * hs, T) -> (B, nh, T, hs)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)

        # self-attention: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q * self.scale) @ k.transpose(-2, -1)
        # prevent q from attending to invalid tokens
        att = att.masked_fill(torch.logical_not(mask[:, :, None, :]), float('-inf'))
        # softmax attn
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        out = att @ (v * mask[:, :, :, None].to(v.dtype))
        # re-assemble all head outputs side by side
        out = out.transpose(2, 3).contiguous().view(B, C, -1)

        # output projection + skip connection
        out = self.proj_drop(self.proj(out)) * mask.to(out.dtype)
        return out, mask


class MaskedMHCA(nn.Module):
    """
    Multi Head Conv Attention with mask

    Add a depthwise convolution within a standard MHA
    The extra conv op can be used to
    (1) encode relative position information (relacing position encoding);
    (2) downsample the features if needed;
    (3) match the feature channels

    Note: With current implementation, the downsampled feature will be aligned
    to every s+1 time step, where s is the downsampling stride. This allows us
    to easily interpolate the corresponding positional embeddings.

    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(
        self,
        n_embd,          # dimension of the output features
        n_head,          # number of heads in multi-head self-attention
        n_qx_stride=1,   # dowsampling stride for query and input
        n_kv_stride=1,   # downsampling stride for key and value
        attn_pdrop=0.0,  # dropout rate for the attention map
        proj_pdrop=0.0,  # dropout rate for projection op
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)

        # conv/pooling operations
        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride

        # query conv (depthwise)
        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.query_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.query_norm = LayerNorm(self.n_embd)

        # key, value conv (depthwise)
        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.key_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.key_norm = LayerNorm(self.n_embd)
        self.value_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.value_norm = LayerNorm(self.n_embd)

        # key, query, value projections for all heads
        # it is OK to ignore masking, as the mask will be attached on the attention
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # query conv -> (B, nh * hs, T')
        q, qx_mask = self.query_conv(x, mask)
        q = self.query_norm(q)
        # key, value conv -> (B, nh * hs, T'')
        k, kv_mask = self.key_conv(x, mask)
        k = self.key_norm(k)
        v, _ = self.value_conv(x, mask)
        v = self.value_norm(v)

        # projections
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        # move head forward to be the batch dim
        # (B, nh * hs, T'/T'') -> (B, nh, T'/T'', hs)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)

        # self-attention: (B, nh, T', hs) x (B, nh, hs, T'') -> (B, nh, T', T'')
        att = (q * self.scale) @ k.transpose(-2, -1)
        # prevent q from attending to invalid tokens
        att = att.masked_fill(torch.logical_not(kv_mask[:, :, None, :]), float('-inf'))
        # softmax attn
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        # (B, nh, T', T'') x (B, nh, T'', hs) -> (B, nh, T', hs)
        out = att @ (v * kv_mask[:, :, :, None].to(v.dtype))
        # re-assemble all head outputs side by side
        out = out.transpose(2, 3).contiguous().view(B, C, -1)

        # output projection + skip connection
        out = self.proj_drop(self.proj(out)) * qx_mask.to(out.dtype)
        return out, qx_mask


class LocalMaskedMHCA(nn.Module):
    """
    Local Multi Head Conv Attention with mask

    Add a depthwise convolution within a standard MHA
    The extra conv op can be used to
    (1) encode relative position information (relacing position encoding);
    (2) downsample the features if needed;
    (3) match the feature channels

    Note: With current implementation, the downsampled feature will be aligned
    to every s+1 time step, where s is the downsampling stride. This allows us
    to easily interpolate the corresponding positional embeddings.

    The implementation is fairly tricky, code reference from
    https://github.com/huggingface/transformers/blob/master/src/transformers/models/longformer/modeling_longformer.py
    """

    def __init__(
        self,
        n_embd,          # dimension of the output features
        n_head,          # number of heads in multi-head self-attention
        window_size,     # size of the local attention window
        n_qx_stride=1,   # dowsampling stride for query and input
        n_kv_stride=1,   # downsampling stride for key and value
        attn_pdrop=0.0,  # dropout rate for the attention map
        proj_pdrop=0.0,  # dropout rate for projection op
        use_rel_pe=False # use relative position encoding
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)
        self.window_size = window_size
        self.window_overlap  = window_size // 2
        # must use an odd window size
        assert self.window_size > 1 and self.n_head >= 1
        self.use_rel_pe = use_rel_pe

        # conv/pooling operations
        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride

        # query conv (depthwise)
        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.query_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.query_norm = LayerNorm(self.n_embd)

        # key, value conv (depthwise)
        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        self.key_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.key_norm = LayerNorm(self.n_embd)
        self.value_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.value_norm = LayerNorm(self.n_embd)

        # key, query, value projections for all heads
        # it is OK to ignore masking, as the mask will be attached on the attention
        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # output projection
        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

        # relative position encoding
        if self.use_rel_pe:
            self.rel_pe = nn.Parameter(
                torch.zeros(1, 1, self.n_head, self.window_size))
            trunc_normal_(self.rel_pe, std=(2.0 / self.n_embd)**0.5)

    @staticmethod
    def _chunk(x, window_overlap):
        """convert into overlapping chunks. Chunk size = 2w, overlap size = w"""
        # x: B x nh, T, hs
        # non-overlapping chunks of size = 2w -> B x nh, T//2w, 2w, hs
        x = x.view(
            x.size(0),
            x.size(1) // (window_overlap * 2),
            window_overlap * 2,
            x.size(2),
        )

        # use `as_strided` to make the chunks overlap with an overlap size = window_overlap
        chunk_size = list(x.size())
        chunk_size[1] = chunk_size[1] * 2 - 1
        chunk_stride = list(x.stride())
        chunk_stride[1] = chunk_stride[1] // 2

        # B x nh, #chunks = T//w - 1, 2w, hs
        return x.as_strided(size=chunk_size, stride=chunk_stride)

    @staticmethod
    def _pad_and_transpose_last_two_dims(x, padding):
        """pads rows and then flips rows and columns"""
        # padding value is not important because it will be overwritten
        x = nn.functional.pad(x, padding)
        x = x.view(*x.size()[:-2], x.size(-1), x.size(-2))
        return x

    @staticmethod
    def _mask_invalid_locations(input_tensor, affected_seq_len):
        beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        beginning_mask = beginning_mask_2d[None, :, None, :]
        ending_mask = beginning_mask.flip(dims=(1, 3))
        beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        beginning_mask = beginning_mask.expand(beginning_input.size())
        # `== 1` converts to bool or uint8
        beginning_input.masked_fill_(beginning_mask == 1, -float("inf"))
        ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        ending_mask = ending_mask.expand(ending_input.size())
        # `== 1` converts to bool or uint8
        ending_input.masked_fill_(ending_mask == 1, -float("inf"))

    @staticmethod
    def _pad_and_diagonalize(x):
        """
        shift every row 1 step right, converting columns into diagonals.
        Example::
              chunked_hidden_states: [ 0.4983,  2.6918, -0.0071,  1.0492,
                                       -1.8348,  0.7672,  0.2986,  0.0285,
                                       -0.7584,  0.4206, -0.0405,  0.1599,
                                       2.0514, -1.1600,  0.5372,  0.2629 ]
              window_overlap = num_rows = 4
             (pad & diagonalize) =>
             [ 0.4983,  2.6918, -0.0071,  1.0492, 0.0000,  0.0000,  0.0000
               0.0000,  -1.8348,  0.7672,  0.2986,  0.0285, 0.0000,  0.0000
               0.0000,  0.0000, -0.7584,  0.4206, -0.0405,  0.1599, 0.0000
               0.0000,  0.0000,  0.0000, 2.0514, -1.1600,  0.5372,  0.2629 ]
        """
        total_num_heads, num_chunks, window_overlap, hidden_dim = x.size()
        # total_num_heads x num_chunks x window_overlap x (hidden_dim+window_overlap+1).
        x = nn.functional.pad(
            x, (0, window_overlap + 1)
        )
        # total_num_heads x num_chunks x window_overlap*window_overlap+window_overlap
        x = x.view(total_num_heads, num_chunks, -1)
        # total_num_heads x num_chunks x window_overlap*window_overlap
        x = x[:, :, :-window_overlap]
        x = x.view(
            total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim
        )
        x = x[:, :, :, :-1]
        return x

    def _sliding_chunks_query_key_matmul(
        self, query, key, num_heads, window_overlap
    ):
        """
        Matrix multiplication of query and key tensors using with a sliding window attention pattern. This implementation splits the input into overlapping chunks of size 2w with an overlap of size w (window_overlap)
        """
        # query / key: B*nh, T, hs
        bnh, seq_len, head_dim = query.size()
        batch_size = bnh // num_heads
        assert seq_len % (window_overlap * 2) == 0
        assert query.size() == key.size()

        chunks_count = seq_len // window_overlap - 1

        # B * num_heads, head_dim, #chunks=(T//w - 1), 2w
        chunk_query = self._chunk(query, window_overlap)
        chunk_key = self._chunk(key, window_overlap)

        # matrix multiplication
        # bcxd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcyd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcxy: batch_size * num_heads x chunks x 2window_overlap x 2window_overlap
        diagonal_chunked_attention_scores = torch.einsum(
            "bcxd,bcyd->bcxy", (chunk_query, chunk_key))

        # convert diagonals into columns
        # B * num_heads, #chunks, 2w, 2w+1
        diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(
            diagonal_chunked_attention_scores, padding=(0, 0, 0, 1)
        )

        # allocate space for the overall attention matrix where the chunks are combined. The last dimension
        # has (window_overlap * 2 + 1) columns. The first (window_overlap) columns are the window_overlap lower triangles (attention from a word to
        # window_overlap previous words). The following column is attention score from each word to itself, then
        # followed by window_overlap columns for the upper triangle.
        diagonal_attention_scores = diagonal_chunked_attention_scores.new_empty(
            (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
        )

        # copy parts from diagonal_chunked_attention_scores into the combined matrix of attentions
        # - copying the main diagonal and the upper triangle
        diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
            :, :, :window_overlap, : window_overlap + 1
        ]
        diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
            :, -1, window_overlap:, : window_overlap + 1
        ]
        # - copying the lower triangle
        diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
            :, :, -(window_overlap + 1) : -1, window_overlap + 1 :
        ]

        diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
            :, 0, : window_overlap - 1, 1 - window_overlap :
        ]

        # separate batch_size and num_heads dimensions again
        diagonal_attention_scores = diagonal_attention_scores.view(
            batch_size, num_heads, seq_len, 2 * window_overlap + 1
        ).transpose(2, 1)

        self._mask_invalid_locations(diagonal_attention_scores, window_overlap)
        return diagonal_attention_scores

    def _sliding_chunks_matmul_attn_probs_value(
        self, attn_probs, value, num_heads, window_overlap
    ):
        """
        Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors. Returned tensor will be of the
        same shape as `attn_probs`
        """
        bnh, seq_len, head_dim = value.size()
        batch_size = bnh // num_heads
        assert seq_len % (window_overlap * 2) == 0
        assert attn_probs.size(3) == 2 * window_overlap + 1
        chunks_count = seq_len // window_overlap - 1
        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size 2 window overlap

        chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
            batch_size * num_heads, seq_len // window_overlap, window_overlap, 2 * window_overlap + 1
        )

        # pad seq_len with w at the beginning of the sequence and another window overlap at the end
        padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)

        # chunk padded_value into chunks of size 3 window overlap and an overlap of size window overlap
        chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)
        chunked_value_stride = padded_value.stride()
        chunked_value_stride = (
            chunked_value_stride[0],
            window_overlap * chunked_value_stride[1],
            chunked_value_stride[1],
            chunked_value_stride[2],
        )
        chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)

        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)

        context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        return context.view(batch_size, num_heads, seq_len, head_dim)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # step 1: depth convolutions
        # query conv -> (B, nh * hs, T')
        q, qx_mask = self.query_conv(x, mask)
        q = self.query_norm(q)
        # key, value conv -> (B, nh * hs, T'')
        k, kv_mask = self.key_conv(x, mask)
        k = self.key_norm(k)
        v, _ = self.value_conv(x, mask)
        v = self.value_norm(v)

        # step 2: query, key, value transforms & reshape
        # projections
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)
        # (B, nh * hs, T) -> (B, nh, T, hs)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        # view as (B * nh, T, hs)
        q = q.view(B * self.n_head, -1, self.n_channels).contiguous()
        k = k.view(B * self.n_head, -1, self.n_channels).contiguous()
        v = v.view(B * self.n_head, -1, self.n_channels).contiguous()

        # step 3: compute local self-attention with rel pe and masking
        q *= self.scale
        # chunked query key attention -> B, T, nh, 2w+1 = window_size
        att = self._sliding_chunks_query_key_matmul(
            q, k, self.n_head, self.window_overlap)

        # rel pe
        if self.use_rel_pe:
            att += self.rel_pe
        # kv_mask -> B, T'', 1
        inverse_kv_mask = torch.logical_not(
            kv_mask[:, :, :, None].view(B, -1, 1))
        # 0 for valid slot, -inf for masked ones
        float_inverse_kv_mask = inverse_kv_mask.type_as(q).masked_fill(
            inverse_kv_mask, -1e4)
        # compute the diagonal mask (for each local window)
        diagonal_mask = self._sliding_chunks_query_key_matmul(
            float_inverse_kv_mask.new_ones(size=float_inverse_kv_mask.size()),
            float_inverse_kv_mask,
            1,
            self.window_overlap
        )
        att += diagonal_mask

        # ignore input masking for now
        att = nn.functional.softmax(att, dim=-1)
        # softmax sometimes inserts NaN if all positions are masked, replace them with 0
        att = att.masked_fill(
            torch.logical_not(kv_mask.squeeze(1)[:, :, None, None]), 0.0)
        att = self.attn_drop(att)

        # step 4: compute attention value product + output projection
        # chunked attn value product -> B, nh, T, hs
        out = self._sliding_chunks_matmul_attn_probs_value(
            att, v, self.n_head, self.window_overlap)
        # transpose to B, nh, hs, T -> B, nh*hs, T
        out = out.transpose(2, 3).contiguous().view(B, C, -1)
        # output projection + skip connection
        out = self.proj_drop(self.proj(out)) * qx_mask.to(out.dtype)
        return out, qx_mask


class TransformerBlock(nn.Module):
    """
    A simple (post layer norm) Transformer block
    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """
    def __init__(
        self,
        n_embd,                # dimension of the input features
        n_head,                # number of attention heads
        n_ds_strides=(1, 1),   # downsampling strides for q & x, k & v
        n_out=None,            # output dimension, if None, set to input dim
        n_hidden=None,         # dimension of the hidden layer in MLP
        act_layer=nn.GELU,     # nonlinear activation used in MLP, default GELU
        attn_pdrop=0.0,        # dropout rate for the attention map
        proj_pdrop=0.0,        # dropout rate for the projection / MLP
        path_pdrop=0.0,        # drop path rate
        mha_win_size=-1,       # > 0 to use window mha
        use_rel_pe=False       # if to add rel position encoding to attention
    ):
        super().__init__()
        assert len(n_ds_strides) == 2
        # layer norm for order (B C T)
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)

        # specify the attention module
        if mha_win_size > 1:
            self.attn = LocalMaskedMHCA(
                n_embd,
                n_head,
                window_size=mha_win_size,
                n_qx_stride=n_ds_strides[0],
                n_kv_stride=n_ds_strides[1],
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                use_rel_pe=use_rel_pe  # only valid for local attention
            )
        else:
            self.attn = MaskedMHCA(
                n_embd,
                n_head,
                n_qx_stride=n_ds_strides[0],
                n_kv_stride=n_ds_strides[1],
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop
            )

        # input
        if n_ds_strides[0] > 1:
            kernel_size, stride, padding = \
                n_ds_strides[0] + 1, n_ds_strides[0], (n_ds_strides[0] + 1)//2
            self.pool_skip = nn.MaxPool1d(
                kernel_size, stride=stride, padding=padding)
        else:
            self.pool_skip = nn.Identity()

        # two layer mlp
        if n_hidden is None:
            n_hidden = 4 * n_embd  # default
        if n_out is None:
            n_out = n_embd
        # ok to use conv1d here with stride=1
        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1),
            act_layer(),
            nn.Dropout(proj_pdrop, inplace=True),
            nn.Conv1d(n_hidden, n_out, 1),
            nn.Dropout(proj_pdrop, inplace=True),
        )

        # drop path
        if path_pdrop > 0.0:
            self.drop_path_attn = AffineDropPath(n_embd, drop_prob = path_pdrop)
            self.drop_path_mlp = AffineDropPath(n_out, drop_prob = path_pdrop)
        else:
            self.drop_path_attn = nn.Identity()
            self.drop_path_mlp = nn.Identity()

    def forward(self, x, mask, pos_embd=None):
        # pre-LN transformer: https://arxiv.org/pdf/2002.04745.pdf
        out, out_mask = self.attn(self.ln1(x), mask)
        out_mask_float = out_mask.to(out.dtype)
        out = self.pool_skip(x) * out_mask_float + self.drop_path_attn(out)
        # FFN
        out = out + self.drop_path_mlp(self.mlp(self.ln2(out)) * out_mask_float)
        # optionally add pos_embd to the output
        if pos_embd is not None:
            out += pos_embd * out_mask_float
        return out, out_mask


class ConvBlock(nn.Module):
    """
    A simple conv block similar to the basic block used in ResNet
    """
    def __init__(
        self,
        n_embd,                # dimension of the input features
        kernel_size=3,         # conv kernel size
        n_ds_stride=1,         # downsampling stride for the current layer
        expansion_factor=2,    # expansion factor of feat dims
        n_out=None,            # output dimension, if None, set to input dim
        act_layer=nn.ReLU,     # nonlinear activation used after conv, default ReLU
    ):
        super().__init__()
        # must use odd sized kernel
        assert (kernel_size % 2 == 1) and (kernel_size > 1)
        padding = kernel_size // 2
        if n_out is None:
            n_out = n_embd

         # 1x3 (strided) -> 1x3 (basic block in resnet)
        width = n_embd * expansion_factor
        self.conv1 = MaskedConv1D(
            n_embd, width, kernel_size, n_ds_stride, padding=padding)
        self.conv2 = MaskedConv1D(
            width, n_out, kernel_size, 1, padding=padding)

        # attach downsampling conv op
        if n_ds_stride > 1:
            # 1x1 strided conv (same as resnet)
            self.downsample = MaskedConv1D(n_embd, n_out, 1, n_ds_stride)
        else:
            self.downsample = None

        self.act = act_layer()

    def forward(self, x, mask, pos_embd=None):
        identity = x
        out, out_mask = self.conv1(x, mask)
        out = self.act(out)
        out, out_mask = self.conv2(out, out_mask)

        # downsampling
        if self.downsample is not None:
            identity, _ = self.downsample(x, mask)

        # residual connection
        out += identity
        out = self.act(out)

        return out, out_mask


# drop path: from https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/common.py
class Scale(nn.Module):
    """
    Multiply the output regression range by a learnable constant value
    """
    def __init__(self, init_value=1.0):
        """
        init_value : initial value for the scalar
        """
        super().__init__()
        self.scale = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32),
            requires_grad=True
        )

    def forward(self, x):
        """
        input -> scale * input
        """
        return x * self.scale
    
class Exp_Scale(nn.Module):
    """
    Multiply the output regression range by a learnable constant value
    """
    def __init__(self, init_value=0.0):
        """
        init_value : initial value for the scalar
        """
        super().__init__()
        self.scale = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32),
            requires_grad=True
        )

    def forward(self, x):
        """
        input -> scale * input
        """
        return self.scale.exp() * x


# The follow code is modified from
# https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/common.py
def drop_path(x, drop_prob=0.0, training=False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AffineDropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks) with a per channel scaling factor (and zero init)
    See: https://arxiv.org/pdf/2103.17239.pdf
    """

    def __init__(self, num_dim, drop_prob=0.0, init_scale_value=1e-4):
        super().__init__()
        self.scale = nn.Parameter(
            init_scale_value * torch.ones((1, num_dim, 1)),
            requires_grad=True
        )
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(self.scale * x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(self, in_features, out_features, num_layers, act_layer=nn.ReLU):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(nn.Linear(in_features, out_features, bias=False))
        else:
            self.layers.append(nn.Linear(in_features, out_features, bias=True))
            if act_layer is not None:
                self.layers.append(act_layer())
            for _ in range(num_layers-2):
                self.layers.append(nn.Linear(out_features, out_features, bias=True))
                if act_layer is not None:
                    self.layers.append(act_layer())
            self.layers.append(nn.Linear(out_features, out_features, bias=False))
        self.layers = nn.Sequential(*self.layers)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.layers(x)
    
class GCNBlock(nn.Module):
    """
    A simple Graph Convolutional Block
    """
    def __init__(
        self,
        n_embd,                # dimension of the input features
        num_node,              # num of node
        num_classes = 43,      # number of object types
        downsample = False,
        scale_factor = 1.0
    ):
        super().__init__()
        feat_dim = n_embd
        self.feat_dim = feat_dim
        self.num_node = num_node
        self.linear = nn.Linear(4, feat_dim, bias=False)
        self.embedding = nn.Embedding(num_classes, feat_dim, padding_idx=0)
        self.relu = nn.ReLU()
        self.conv2d_1_weight = nn.Conv2d(feat_dim * 2, feat_dim, 1, 1, 0, bias=False)
        self.conv2d_2_weight = nn.Conv2d(feat_dim, feat_dim, 1, 1, 0, bias=False)
        self.conv1d_merge_node_weight = nn.Conv1d(feat_dim, feat_dim, num_node, 1, 0, bias=False)
        self.layernorm = LayerNorm(self.feat_dim)
        self.downsample = downsample
        self.scale_factor = scale_factor

    def generate_graph_feature(self, bbox, bbox_class, edge_map):
        B, T, N, C = bbox_class.size()
        node = torch.cat((bbox, bbox_class), dim=3).permute(0, 3, 1, 2) # becomes (B, C, T, N)
        node = self.conv2d_1_weight(node).permute(0, 2, 3, 1) # becomes (B, T, N, C)
        node = torch.einsum('btij,btjk->btik', edge_map, node)
        node = self.relu(node).permute(0, 3, 1, 2) # becomes (B, C, T, N)
        node = self.conv2d_2_weight(node).permute(0, 2, 3, 1) # becomes (B, T, N, C)
        node = torch.einsum('btij,btjk->btik', edge_map, node)
        node = self.relu(node)
        node = node.permute(0, 1, 3, 2).reshape(-1, self.feat_dim, self.num_node)
        node = self.conv1d_merge_node_weight(node)
        return node.reshape(B, T, self.feat_dim)
    
    def forward(self, bbox, bbox_class, edge_map):
        bbox = bbox.permute(0, 3, 1, 2) # B, T, N, 4
        bbox_class = bbox_class.permute(0, 2, 1) # B, T, N
        edge_map = edge_map.permute(0, 3, 1, 2) # B, T, N, N
        bbox_class_embed = self.embedding(bbox_class)
        bbox_embed = self.linear(bbox)
        feature = self.generate_graph_feature(bbox_class_embed, bbox_embed, edge_map)

        if self.downsample:
            return self.layernorm(F.interpolate(feature.permute(0, 2, 1), scale_factor=self.scale_factor, mode='linear'))
        else:
            return self.layernorm(feature.permute(0, 2, 1))
        

class DynELayer(nn.Module):
    def __init__(
            self,
            n_embd,  # dimension of the input features
            kernel_size=3,  # conv kernel size
            n_ds_stride=1,  # downsampling stride for the current layer
            k=1.5,  # k
            group=1,  # group for cnn
            n_out=None,  # output dimension, if None, set to input dim
            n_hidden=None,  # hidden dim for mlp
            path_pdrop=0.0,  # drop path rate
            act_layer=nn.GELU,  # nonlinear activation used in mlp,
            init_conv_vars=0.1  # init gaussian variance for the weight
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = n_ds_stride
        if n_out is None:
            n_out = n_embd

        self.ln = LayerNorm(n_embd)
        self.gn = nn.GroupNorm(16, n_embd)

        assert kernel_size % 2 == 1
        # add 1 to avoid have the same size as the instant-level branch
        up_size = round((kernel_size + 1) * k)
        up_size = up_size + 1 if up_size % 2 == 0 else up_size

        self.psi    = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)
        self.convw  = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)
        self.convkw = nn.Conv1d(n_embd, n_embd, up_size, stride=1, padding=up_size // 2, groups=n_embd)

        
        self.fc        = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)
        self.global_fc = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)

        if n_ds_stride > 1:
                kernel_size, stride, padding = \
                    n_ds_stride + 1, n_ds_stride, (n_ds_stride + 1) // 2
                self.downsample = nn.MaxPool1d(
                    kernel_size, stride=stride, padding=padding)
                self.stride = stride
        else:
            self.downsample = nn.Identity()
            self.stride = 1

        # two layer mlp
        if n_hidden is None:
            n_hidden = 4 * n_embd  # default
        if n_out is None:
            n_out = n_embd

        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1, groups=group),
            act_layer(),
            nn.Conv1d(n_hidden, n_out, 1, groups=group),
        )

        # drop path
        if path_pdrop > 0.0:
            self.drop_path_out = AffineDropPath(n_embd, drop_prob=path_pdrop)
            self.drop_path_mlp = AffineDropPath(n_out, drop_prob=path_pdrop)
        else:
            self.drop_path_out = nn.Identity()
            self.drop_path_mlp = nn.Identity()

        self.act = act_layer()
        self.reset_params(init_conv_vars=init_conv_vars)

    def reset_params(self, init_conv_vars=0):

        torch.nn.init.normal_(self.psi.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.fc.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convw.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convkw.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.global_fc.weight, 0, init_conv_vars)
        torch.nn.init.constant_(self.psi.bias, 0)
        torch.nn.init.constant_(self.fc.bias, 0)
        torch.nn.init.constant_(self.convw.bias, 0)
        torch.nn.init.constant_(self.convkw.bias, 0)
        torch.nn.init.constant_(self.global_fc.bias, 0)

    def forward(self, x, mask):
        # X shape: B, C, T
        B, C, T = x.shape
        x = self.downsample(x)
        out_mask = F.interpolate(
            mask.to(x.dtype),
            size=torch.div(T, self.stride, rounding_mode='trunc'),
            mode='nearest'
        ).detach()

        out = self.ln(x)
        psi = self.psi(out)
        fc = self.fc(out)

        phi = torch.relu(self.global_fc(out.mean(dim=-1, keepdim=True)))

        convw = self.convw(out)
        convkw = self.convkw(out)

        out = fc * phi + torch.relu(convw + convkw) * psi + out
        
        out = x * out_mask + self.drop_path_out(out)
        # FFN
        out = out + self.drop_path_mlp(self.mlp(self.gn(out)))

        return out, out_mask.bool()
    
class DynamicConv1D_chk(nn.Module):
    def __init__(
        self, 
        in_channels : int,
        out_channels : int,
        kernel_size : int = 1,
        padding : int = 0,
        stride : int = 1,
        num_groups : int = 1,
        norm: str = "LN",
        gate_activation : str = "ReTanH",
        gate_activation_kargs : dict = None
    ):
        super(DynamicConv1D_chk, self).__init__()
        
        self.num_groups = num_groups
        self.norm = norm
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        convs = []

        convs += [nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                           stride=stride, padding=padding, groups= num_groups),
                    LayerNorm(out_channels)]
        in_channels = out_channels

        self.convs = nn.Sequential(*convs)
        self.gate = TemporalGate(self.in_channels,
                                #  out_dim=num_groups,
                                num_groups=num_groups,
                                kernel_size=kernel_size,
                                padding=padding,
                                stride=stride,
                                gate_activation=gate_activation,
                                gate_activation_kargs = gate_activation_kargs)

    def get_running_cost(self, gate):
        
        conv_cost = self.in_channels * self.out_channels * len(self.convs) * \
                self.kernel_size
        norm_cost = self.out_channels if self.norm != "none" else 0
        unit_cost = conv_cost + norm_cost

        hard_gate = (gate != 0).float()
        cost = [gate.detach() * unit_cost / self.num_groups,
                hard_gate * unit_cost / self.num_groups,
                torch.ones_like(gate) * unit_cost / self.num_groups]

        cost = [x.flatten(1).sum(-1) for x in cost]
        
        # print(cost[0]/cost[2], cost[1]/cost[2])
        
        return cost

    def forward(self, input, mask):

        out_mask = mask.to(input.dtype)
        data = self.convs(input)
        data = data * out_mask.detach()
        output = self.gate(data, input, out_mask)
        # masking the output, stop grad to mask
        output = output * out_mask.detach()
        out_mask = out_mask.bool()

        return output, out_mask

class DynamicScale_chk(nn.Module):
    def __init__(
        self,
        in_channels : int,
        out_channels : int,
        num_convs: int = 1,
        kernel_size : int = 1,
        padding : int = 0,
        stride : int = 1,
        num_groups : int = 1,
        num_adjacent_scales: int = 2,
        depth_module: nn.Module = None,
        norm: str = "GN",
        gate_activation : str = "ReTanH",
        gate_activation_kargs : dict = None
    ):
        super(DynamicScale_chk, self).__init__()
        self.num_groups = num_groups
        self.num_adjacent_scales = num_adjacent_scales
        self.depth_module = depth_module
        dynamic_convs = [
            DTFAM(dim=in_channels, o_dim= in_channels, ka=kernel_size, gate_activation="GeReTanH",
                gate_activation_kargs = gate_activation_kargs)
            
 
        for _ in range(num_adjacent_scales)]
        
        self.dynamic_convs = nn.ModuleList(dynamic_convs)
        
        self.resize = lambda x, s : F.interpolate(
            x, size=s, mode="nearest")
        
        self.scale_weight = nn.Parameter(torch.zeros(1))
        self.output_weight = nn.Parameter(torch.ones(1))
        self.init_parameters()

        self.norm = LayerNorm(out_channels)
        self.act = nn.ReLU()

    def init_parameters(self):
        for module in self.dynamic_convs:
            module.init_parameters()

    def forward(self, inputs, fpn_masks):

        dynamic_scales = []
        for l, x in enumerate(inputs):
            dynamic_scales.append([m(x, fpn_masks[l])[0] for m in self.dynamic_convs])
        
        outputs = []
        out_masks = []
        for l, x in enumerate(inputs):
            scale_feature = []
            
            for s in range(self.num_adjacent_scales):
                l_source = l + s - self.num_adjacent_scales // 2
                l_source = l_source if l_source < l else l_source + 1
                if l_source >= 0 and l_source < len(inputs):
                    
                    feature = self.resize(dynamic_scales[l_source][s], x.shape[-1:])
                    scale_feature.append(feature)
                         
            scale_feature = sum(scale_feature) * self.scale_weight + x * self.output_weight
            
            if self.depth_module is not None:
                scale_feature, masks = self.depth_module(scale_feature, fpn_masks[l])

            outputs.append(scale_feature)
                 
        out_masks = fpn_masks
        
        return outputs, out_masks

class TemporalGate(nn.Module):
    def __init__(
        self,
        in_channels : int,
        num_groups : int = 1,
        kernel_size : int = 1,
        padding : int = 0,
        stride : int = 1,
        attn_gate : bool = False,
        gate_activation : str = "ReTanH",
        gate_activation_kargs : dict = None,
        head_gate = True
    ):
        super(TemporalGate, self).__init__()
        self.num_groups = num_groups
        self.in_channels = in_channels
        self.head_gate = head_gate
        self.ka = kernel_size
        
        if num_groups == kernel_size:
            self.gate_conv = nn.Conv1d(in_channels=in_channels, out_channels=num_groups, kernel_size=kernel_size,
                           stride=stride, padding=padding)
        elif num_groups == kernel_size*in_channels:
            self.gate_conv = nn.Conv1d(in_channels=in_channels, out_channels=num_groups, kernel_size=kernel_size,
                           stride=stride, padding=padding, groups=in_channels)
        else:
            self.gate_conv = nn.Conv1d(in_channels=in_channels, out_channels=num_groups, kernel_size=kernel_size,
                           stride=stride, padding=padding, groups=num_groups)
        
        self.gate_activation = gate_activation
        self.gate_activation_kargs = gate_activation_kargs
        if gate_activation == "ReTanH":
            self.gate_activate = lambda x : torch.tanh(x).clamp(min=0)

        elif gate_activation == "ReLU":
            self.gate_activate = lambda x : torch.relu(x)

        elif gate_activation == "Sigmoid":
            self.gate_activate = lambda x : torch.sigmoid(x)

        elif gate_activation == "GeReTanH":
            assert "tau" in gate_activation_kargs
            tau = gate_activation_kargs["tau"]
            ttau = math.tanh(tau)
            self.gate_activate = lambda x : ((torch.tanh(x - tau) + ttau) / (1 + ttau)).clamp(min=0)
        else:
            raise NotImplementedError()

    def encode(self, *inputs):

        if self.num_groups == self.ka * self.in_channels:
            return inputs
        
        if self.num_groups == self.ka:
            da, mask = inputs
            b,ck,t = inputs[0].shape
            x = inputs[0].view(b, self.in_channels, self.ka, t)
            da = x.permute(0,2,1,3).contiguous().view(b, ck, t)
            inputs = (da,mask)
        
        outputs = [x.view(x.shape[0] * self.num_groups, -1, *x.shape[2:]) for x in inputs]

        return outputs

    def decode(self, *inputs):
        if self.num_groups == self.ka * self.in_channels:
            return inputs

        outputs = [x.view(x.shape[0] // self.num_groups, -1, *x.shape[2:]) for x in inputs]
        return outputs

    def forward(self, data_input, gate_input, mask):
        # data_input b c h w

        out_mask = mask.to(data_input.dtype)
        
        data = data_input * out_mask.detach()
        gate = self.gate_conv(gate_input)
        gate = self.gate_activate(gate)
        gate = gate*out_mask

        data, gate = self.encode(data_input, gate)
        output, = self.decode(data * gate)
        return output

class DTFAM(nn.Module):    
    def __init__(self, dim= 512, o_dim = 1, ka=3, stride=1, groups = 1, padding_mode='zeros', conv_type= 'gate', gate_activation : str = "ReTanH",
        gate_activation_kargs : dict = None):
        super().__init__()
        
        self.dim = dim

        self.padding_mode = padding_mode
        
        self.ka = ka
        self.stride = stride

        self.shift_conv = nn.Conv1d(dim, dim*ka, kernel_size=self.ka, stride= stride, bias=False, groups= dim, padding=self.ka//2, padding_mode=padding_mode)
        self.conv = nn.Conv1d(dim*ka, o_dim, kernel_size=1, bias=True, groups = groups, padding=0)

        dyn_type = gate_activation_kargs['dyn_type']
        self.conv_type = conv_type
        if self.conv_type == 'gate':
            if dyn_type == 'c':
                self.kernel_conv = TemporalGate(dim,
                                num_groups=dim,
                                kernel_size=ka,
                                padding=ka//2,
                                stride=1,
                                gate_activation=gate_activation,
                                gate_activation_kargs = gate_activation_kargs)
            elif dyn_type == 'k':
                self.kernel_conv = TemporalGate(dim,
                                num_groups=ka,
                                kernel_size=ka,
                                padding=ka//2,
                                stride=1,
                                gate_activation=gate_activation,
                                gate_activation_kargs = gate_activation_kargs)
            
            elif dyn_type == 'ck':
                self.kernel_conv = TemporalGate(dim,
                                num_groups=dim*ka,
                                kernel_size=ka,
                                padding=ka//2,
                                stride=1,
                                gate_activation=gate_activation,
                                gate_activation_kargs = gate_activation_kargs)
            else:
                assert 1==0
        else:
            self.kernel_conv = DynamicConv1D_chk(
            in_channels = dim*self.ka,
            out_channels = dim,
            kernel_size=self.ka,
            padding=self.ka//2,
            stride=stride,
            num_groups=groups,
            gate_activation=gate_activation,
            gate_activation_kargs=gate_activation_kargs)

        self.norm = LayerNorm(o_dim)
        
        self.init_parameters()
        

    def shift(self, x):
        # Pure shift operation, we do not use this operation in this repo.
        # We use constant kernel conv for shift.
        B, C, T = x.shape
        
        out = torch.zeros((B,self.ka*C, T), device=x.device)
        padx = F.pad(x,(self.ka//2,self.ka//2))

        for i in range(self.ka):
            out[:, i*C:(i+1)*C, : ] = padx[:, :, i:i+T]
        
        out = out.reshape(B, self.ka ,C , T)
        out = torch.transpose(out, 1,2) 
        out = out.reshape(B, self.ka* C , T)
        
        return out
    
    def init_parameters(self):
        #  shift initialization for group convolution
        kernel = torch.zeros(self.ka, 1, self.ka)
        for i in range(self.ka):
            kernel[i, 0, i] = 1.

        kernel = kernel.repeat(self.dim, 1, 1)
        self.shift_conv.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x, mask):
        B, C, T = x.shape

        _x = self.shift_conv(x)
        
        if self.conv_type == 'gate':
            weight = self.kernel_conv(_x, x, mask)
        else:
            weight, _ = self.kernel_conv(_x, mask) 
            weight = weight.repeat_interleave(self.ka, dim = 1)
        _x = _x*weight
        
        out_conv = self.conv(_x)        
        out_conv = self.norm(out_conv)
        if self.stride > 1:
            # downsample the mask using nearest neighbor
            out_mask = F.interpolate(
                mask.to(x.dtype), size=out_conv.size(-1), mode='nearest' )
        else:
            out_mask = mask.to(x.dtype)

        out_conv = out_conv * out_mask.detach()
        out_mask = out_mask.bool()
        
        return out_conv, out_mask

class DynamicFeatureAttentionLayer(nn.Module):
    """
    Simplified Dynamic Temporal Feature Attention Module
    
    A lightweight implementation inspired by DTFAM that applies dynamic attention
    to temporal features using learnable gates.
    """
    
    def __init__(self, dim=256, kernel_size=3, reduction_ratio=4):
        """
        Args:
            dim: Input feature dimension
            kernel_size: Size of temporal convolution kernel
            reduction_ratio: Channel reduction ratio for efficiency
        """
        super().__init__()
        
        self.dim = dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # Temporal shift convolution (similar to original shift_conv)
        self.temporal_conv = nn.Conv1d(
            dim, dim * kernel_size, 
            kernel_size=kernel_size,
            padding=self.padding,
            groups=dim,  # Depthwise convolution
            bias=False
        )
        
        # Dynamic gate generation network
        gate_dim = max(dim // reduction_ratio, 1)
        self.gate_network = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Global temporal pooling
            nn.Conv1d(dim, gate_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(gate_dim, dim * kernel_size, 1),
            nn.Sigmoid()  # Gate values between 0 and 1
        )
        
        # Output projection
        self.output_conv = nn.Conv1d(dim * kernel_size, dim, 1)
        self.norm = nn.LayerNorm(dim)
        
        # Initialize temporal convolution for identity mapping
        self._init_temporal_conv()
    
    def _init_temporal_conv(self):
        """Initialize temporal convolution to perform identity + shift operations"""
        with torch.no_grad():
            # Create identity kernel for each shift position
            weight = torch.zeros(self.dim * self.kernel_size, 1, self.kernel_size)
            for i in range(self.kernel_size):
                for j in range(self.dim):
                    weight[i * self.dim + j, 0, i] = 1.0
            self.temporal_conv.weight.copy_(weight)
    
    def forward(self, x, mask=None):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape [B, C, T]
            mask: Optional mask tensor of shape [B, 1, T]
            
        Returns:
            output: Dynamic feature tensor [B, C, T]
            mask: Updated mask if provided
        """
        B, C, T = x.shape
        
        # Apply temporal convolution to create shifted features
        # [B, C, T] -> [B, C*K, T] where K is kernel_size
        shifted_features = self.temporal_conv(x)
        
        # Generate dynamic gates based on global context
        # [B, C, T] -> [B, C*K, 1] -> [B, C*K, T]
        gates = self.gate_network(x)
        gates = gates.expand(-1, -1, T)
        
        # Apply dynamic gating to shifted features
        gated_features = shifted_features * gates
        
        # Project back to original dimension
        # [B, C*K, T] -> [B, C, T]
        output = self.output_conv(gated_features)
        
        # Apply mask if provided
        if mask is not None:
            output = output * mask.to(output.dtype)
        
        # Layer normalization (applied along channel dimension)
        # Convert to [B, T, C] for LayerNorm, then back to [B, C, T]
        output = output.transpose(1, 2)  # [B, C, T] -> [B, T, C]
        output = self.norm(output)
        output = output.transpose(1, 2)  # [B, T, C] -> [B, C, T]
        
        # Residual connection
        output = output + x
        
        return output, mask
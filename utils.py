import numpy as np
import torch
from scipy.stats import rankdata
import copy
from scipy.linalg import expm


def recall_at_k(gt_mat, results, k=10):
    recall_sum = 0
    for i in range(gt_mat.shape[0]):
        relevant_items = set(np.where(gt_mat[i, :] > 0)[0])
        top_predicted_items = np.argsort(-results[i, :])[:k]
        num_relevant_items = len(relevant_items.intersection(top_predicted_items))
        recall_sum += num_relevant_items / len(relevant_items)
    recall = recall_sum / gt_mat.shape[0]
    return recall


def ndcg_at_k(gt_mat, results, k=10):
    ndcg_sum = 0
    for i in range(gt_mat.shape[0]):
        relevant_items = set(np.where(gt_mat[i, :] > 0)[0])
        top_predicted_items = np.argsort(-results[i, :])[:k]
        dcg = 0
        idcg = 0
        for j in range(k):
            if top_predicted_items[j] in relevant_items:
                dcg += 1 / np.log2(j + 2)
            if j < len(relevant_items):
                idcg += 1 / np.log2(j + 2)
        ndcg_sum += dcg / idcg if idcg > 0 else 0
    ndcg = ndcg_sum / gt_mat.shape[0]
    return ndcg


def calculate_row_correlations(matrix1, matrix2):
    base_value = 1  

    num_rows = matrix1.shape[0]
    correlations = torch.zeros(num_rows)

    for row in range(num_rows):
        nz_indices1 = matrix1.indices[matrix1.indptr[row] : matrix1.indptr[row + 1]]
        nz_indices2 = matrix2.indices[matrix2.indptr[row] : matrix2.indptr[row + 1]]

        common_indices = torch.intersect1d(nz_indices1, nz_indices2)

        nz_values1 = matrix1.data[matrix1.indptr[row] : matrix1.indptr[row + 1]][
            torch.searchsorted(nz_indices1, common_indices)
        ]
        nz_values2 = matrix2.data[matrix2.indptr[row] : matrix2.indptr[row + 1]][
            torch.searchsorted(nz_indices2, common_indices)
        ]

        if len(common_indices) > 0:
            correlation = torch.corrcoef(nz_values1, nz_values2)[0, 1]
            correlations[row] = correlation + base_value

    return correlations

# Helper function to convert CSR to Torch tensor
def csr2torch(csr_matrix):
    coo = csr_matrix.tocoo()
    return torch.sparse_coo_tensor(
        torch.LongTensor([coo.row, coo.col]),
        torch.FloatTensor(coo.data),
        torch.Size(coo.shape),
    ).to_dense()



def csr2torch_(csr_matrix):
    # Convert CSR matrix to COO format (Coordinate List)
    coo_matrix = csr_matrix.tocoo()

    # Create a PyTorch tensor for data, row indices, and column indices
    data = torch.FloatTensor(coo_matrix.data)
    # indices = torch.LongTensor([coo_matrix.row, coo_matrix.col])
    # -> This results in: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor.
    indices = torch.LongTensor(np.vstack((coo_matrix.row, coo_matrix.col)))

    # Create a sparse tensor using torch.sparse
    # return torch.sparse.FloatTensor(indices, data, torch.Size(coo_matrix.shape))
    # -> This results in: UserWarning: torch.sparse.SparseTensor(indices, values, shape, *, device=) is deprecated.  Please use torch.sparse_coo_tensor(indices, values, shape, dtype=, device=).
    return torch.sparse_coo_tensor(indices, data, torch.Size(coo_matrix.shape))


def min_max_normalize(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized = (tensor - min_val) / (max_val - min_val)
    return normalized



def normalize_sparse_adjacency_matrix(adj_matrix, alpha):
    # Calculate rowsum and columnsum using COO format
    rowsum = torch.sparse.mm(
        adj_matrix, torch.ones((adj_matrix.shape[1], 1), device=adj_matrix.device)
    ).squeeze()
    rowsum = torch.pow(rowsum, -alpha)
    colsum = torch.sparse.mm(
        adj_matrix.t(), torch.ones((adj_matrix.shape[0], 1), device=adj_matrix.device)
    ).squeeze()
    colsum = torch.pow(colsum, alpha - 1)
    indices = (
        torch.arange(0, rowsum.size(0)).unsqueeze(1).repeat(1, 2).to(adj_matrix.device)
    )
    # d_mat_rows = torch.sparse.FloatTensor(
    #     indices.t(), rowsum, torch.Size([rowsum.size(0), rowsum.size(0)])
    # ).to(device=adj_matrix.device)
    # -> This results in:  UserWarning: torch.sparse.SparseTensor(indices, values, shape, *, device=) is deprecated.  Please use torch.sparse_coo_tensor(indices, values, shape, dtype=, device=).
    d_mat_rows = torch.sparse_coo_tensor(
        indices.t(), rowsum, torch.Size([rowsum.size(0), rowsum.size(0)])
    ).to(device=adj_matrix.device)
    indices = (
        torch.arange(0, colsum.size(0)).unsqueeze(1).repeat(1, 2).to(adj_matrix.device)
    )
    # d_mat_cols = torch.sparse.FloatTensor(
    # indices.t(), colsum, torch.Size([colsum.size(0), colsum.size(0)])
    # ).to(device=adj_matrix.device)
    # -> This results in: UserWarning: torch.sparse.SparseTensor(indices, values, shape, *, device=) is deprecated.  Please use torch.sparse_coo_tensor(indices, values, shape, dtype=, device=).
    d_mat_cols = torch.sparse_coo_tensor(
        indices.t(), colsum, torch.Size([colsum.size(0), colsum.size(0)])
    ).to(device=adj_matrix.device)

    # Calculate d_inv for rows and columns
    # d_inv_rows = torch.pow(rowsum, -alpha)
    # d_inv_rows[d_inv_rows == float('inf')] = 0.
    # d_mat_rows = torch.diag(d_inv_rows)

    # d_inv_cols = torch.pow(colsum, alpha - 1)
    # d_inv_cols[d_inv_cols == float('inf')] = 0.
    # d_mat_cols = torch.diag(d_inv_cols)

    # Normalize adjacency matrix
    norm_adj = d_mat_rows.mm(adj_matrix).mm(d_mat_cols)

    return norm_adj



def freq_filter(s_values, mode=1, alpha=0.9, start=0):
    """
    input:
    - s_values: singular (eigen) values, list form

    output:
    - filterd_s_values
    """
    if mode == 0:
        filtered_s_values = s_values
    elif mode == 1:
        filtered_s_values = [(lambda x: 1 / (1 - alpha * x))(v) for v in s_values]
    elif mode == 2:
        filtered_s_values = [(lambda x: 1 / (alpha * x))(v) for v in s_values]
    elif mode == 3:
        filtered_s_values = [(lambda x: 1.5**x)(v) for v in s_values]
    elif mode == 3:
        filtered_s_values = [(lambda x: 1.5**x)(v) for v in s_values]
    elif mode == "band_pass":
        end = start + 5
        filtered_s_values = (
            [0] * int(start) + [1] * int(end - start) + [0] * int(len(s_values) - end)
        )

    return np.diag(filtered_s_values)


def get_norm_adj(alpha, adj_mat):
    # Calculate rowsum and columnsum using PyTorch operations
    rowsum = torch.sum(adj_mat, dim=1)
    colsum = torch.sum(adj_mat, dim=0)

    # Calculate d_inv for rows and columns
    d_inv_rows = torch.pow(rowsum, -alpha).flatten()
    d_inv_rows[torch.isinf(d_inv_rows)] = 0.0
    d_mat_rows = torch.diag(d_inv_rows)

    d_inv_cols = torch.pow(colsum, alpha - 1).flatten()
    d_inv_cols[torch.isinf(d_inv_cols)] = 0.0
    d_mat_cols = torch.diag(d_inv_cols)
    d_mat_i_inv_cols = torch.diag(1 / d_inv_cols)

    # Normalize adjacency matrix
    norm_adj = adj_mat.mm(d_mat_rows).mm(adj_mat).mm(d_mat_cols)
    norm_adj = norm_adj.to_sparse()  # Convert to sparse tensor

    # Convert d_mat_rows, d_mat_i_inv_cols to sparse tensors
    d_mat_rows_sparse = d_mat_rows.to_sparse()
    d_mat_i_inv_cols_sparse = d_mat_i_inv_cols.to_sparse()

    return norm_adj


def top_k(S, k=1, device="cpu"):
    """
    S: scores, numpy array of shape (M, N) where M is the number of source nodes,
        N is the number of target nodes
    k: number of predicted elements to return
    """
    if device == "cpu":
        top = torch.argsort(-S)[:, :k]
        result = torch.zeros(S.shape)
        for idx, target_elms in enumerate(top):
            for elm in target_elms:
                result[idx, elm] = 1
    else:
        top = torch.argsort(-S)[:, :k]
        result = torch.zeros(S.shape, device=device)
        for idx, target_elms in enumerate(top):
            for elm in target_elms:
                result[idx, elm] = 1
    return result, top



def precision_k(topk, gt, k, device="cpu"):
    """
    topk, gt: (U, X, I) array, where U is the number of users, X is the number of items per user, and I is the number of items in total.
    k: @k measurement
    """
    if device == "cpu":
        precision_values = []
        for i in range(topk.shape[0]):
            num_correct = np.multiply(topk[i], gt[i]).sum()
            precision_i = num_correct / (k)
            precision_values.append(precision_i)
        mean_precision = np.mean(precision_values)
    else:
        precision_values = []
        for i in range(topk.shape[0]):
            num_correct = torch.mul(topk[i], gt[i]).sum()
            precision_i = num_correct / (k)
            precision_values.append(precision_i)
        mean_precision = torch.mean(torch.tensor(precision_values))

    return mean_precision



def recall_k(topk, gt, k, device="cpu"):
    """
    topk, gt: (U, X, I) array, where U is the number of users, X is the number of items per user, and I is the number of items in total.
    k: @k measurement
    """
    if device == "cpu":
        recall_values = []
        for i in range(topk.shape[0]):
            recall_i = (
                np.multiply(topk[i], gt[i]).sum() / gt[i].sum()
                if gt[i].sum() != 0
                else 0
            )
            if gt[i].sum() != 0:
                recall_values.append(recall_i)
        mean_recall = np.mean(recall_values)
    else:
        recall_values = []
        for i in range(topk.shape[0]):
            recall_i = (
                torch.mul(topk[i], gt[i]).sum() / gt[i].sum() if gt[i].sum() != 0 else 0
            )
            if gt[i].sum() != 0:
                recall_values.append(recall_i)
        mean_recall = torch.mean(torch.tensor(recall_values))

    return mean_recall


def ndcg_k(rels, rels_ideal, gt, device="cpu"):
    """
    rels: sorted top-k arr
    rels_ideal: sorted top-k ideal arr
    """
    k = rels.shape[1]
    n = rels.shape[0]

    ndcg_values = []
    for row in range(n):
        dcg = 0
        idcg = 0
        for col in range(k):
            if gt[row, rels[row, col]] == 1:
                if col == 0:
                    dcg += 1
                else:
                    dcg += 1 / np.log2(col + 1)
            if gt[row, rels_ideal[row, col]] == 1:
                if col == 0:
                    idcg += 1
                else:
                    idcg += 1 / np.log2(col + 1)
        if idcg != 0:
            ndcg_values.append(dcg / idcg)

    mean_ndcg = torch.mean(torch.tensor(ndcg_values))

    return mean_ndcg

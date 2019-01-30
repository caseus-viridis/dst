import torch
import torch.nn as nn
from .structured_sparse import StructuredSparseParameter


param_count = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad) if isinstance(m, nn.Module) else 0

def get_sparse_param_stats(model):
    breakdown = {n: m.param_count() for n, m in model.named_modules() if isinstance(m, StructuredSparseParameter)}
    n_total = param_count(model)
    n_nonzero = sum([_n for _, (_n, _) in breakdown.items()])
    n_sparse = sum([_n for _, (_, _n) in breakdown.items()])
    n_dense = n_total - n_sparse
    sparsity = float(n_sparse - n_nonzero) / n_total
    return n_total, n_dense, n_sparse, n_nonzero, sparsity, breakdown

def prune_by_threshold(model, threshold, p=1):
    # prune or all sparse parameters by a global threshold
    return {
        n: m.prune_by_threshold(threshold, p=p)
        for n, m in model.named_modules()
        if isinstance(m, StructuredSparseParameter)
    }

def prune_or_grow_to_sparsity(model, sparsity, p=1):
    # prune or grow all sparse parameters to a target sparsity
    return {
        n: m.prune_or_grow_to_sparsity(sparsity, p=p)
        for n, m in model.named_modules()
        if isinstance(m, StructuredSparseParameter)
    }

def adjust_pruning_threshold(old_threshold, num_pruned, target_num_pruned, tolerance=0.1, gain=2.):
    # TODO
    if (num_pruned/target_num_pruned):
        pass
    return new_threshold


class DSNetwork(nn.Module):
    r"""
    A dynamic sparse network container for models with dynamic sparse layers
    """
    def __init__(self):
        super(DSNetwork, self).__init__()


    # def prune(self,prune_fraction_fc,prune_fraction_conv,prune_fraction_fc_special = None):
    #     for x in [x for x  in self.modules() if isinstance(x,SparseTensor)]:
    #         if x.conv_tensor:
    #             x.prune_small_connections(prune_fraction_conv)
    #         else:
    #             if x.s_tensor.size(0) == 10 and  x.s_tensor.size(1) == 100:
    #                 x.prune_small_connections(prune_fraction_fc_special)
    #             else:
    #                 x.prune_small_connections(prune_fraction_fc)


    # def get_model_size(self):
    #     def get_tensors_and_test(tensor_type):
    #         relevant_tensors = [x for x in self.modules() if isinstance(x,tensor_type)]
    #         relevant_params = [p for x in relevant_tensors for p in x.parameters()]
    #         is_relevant_param = lambda x : [y for y in relevant_params if x is y]

    #         return relevant_tensors,is_relevant_param

    #     sparse_tensors,is_sparse_param = get_tensors_and_test(SparseTensor)
    #     tied_tensors,is_tied_param = get_tensors_and_test(TiedTensor)


    #     sparse_params = [p for x in sparse_tensors for p in x.parameters()]
    #     is_sparse_param = lambda x : [y for y in sparse_params if x is y]


    #     sparse_size = sum([x.get_sparsity()[0].item() for x in sparse_tensors])

    #     tied_size = 0
    #     for k in tied_tensors:
    #         unique_reps = k.weight_alloc.cpu().unique()
    #         subtensor_size = np.prod(list(k.bank.size())[1:])
    #         tied_size += unique_reps.size(0) * subtensor_size


    #     fixed_size = sum([p.data.nelement()  for p in self.parameters() if (not is_sparse_param(p) and not is_tied_param(p))])
    #     model_size = {'sparse': sparse_size,'tied' : tied_size, 'fixed':fixed_size,'learnable':fixed_size + sparse_size + tied_size}
    #     return model_size

import torch
import torch.nn as nn

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

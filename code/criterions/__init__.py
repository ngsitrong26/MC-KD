from .cross_entropy_loss import CrossEntropyLoss
from .various_divergence import VariousDivergence
from .dual_space_kd import DualSpaceKD
from .dual_space_kd_new import DualSpaceKDWithCMA_OT
from .dual_space_kd_with_cross_model_attention import DualSpaceKDWithCMA
from .universal_logit_distillation import UniversalLogitDistillation
from .min_edit_dis_kld import MinEditDisForwardKLD
from .dual_space_kd_new_kb1 import DualSpaceKDWithCMA_OT_1
from .dual_space_kd_new_kb2 import DualSpaceKDWithCMA_OT_2
from .ULD_1 import UniversalLogitDistillation_1
from .MinED_1 import MinEditDisForwardKLD_1
from .MultiLevelOT import MultiLevelOT
from .MultiLevelOT_1 import MultiLevelOT_1

criterion_list = {
    "cross_entropy": CrossEntropyLoss,
    "various_divergence": VariousDivergence,
    "dual_space_kd": DualSpaceKD,
    "dual_space_kd_with_cma": DualSpaceKDWithCMA,
    "universal_logit_distillation": UniversalLogitDistillation,
    "min_edit_dis_kld": MinEditDisForwardKLD,
    "dual_space_kd_with_cma_ot": DualSpaceKDWithCMA_OT,
    "dual_space_kd_with_cma_ot_kb1": DualSpaceKDWithCMA_OT_1,
    "dual_space_kd_with_cma_ot_kb2": DualSpaceKDWithCMA_OT_2,
    "uld_1": UniversalLogitDistillation_1,
    "mined_1": MinEditDisForwardKLD_1,
    "MultiLevelOT": MultiLevelOT,
    "MultiLevelOT_1": MultiLevelOT_1
}

def build_criterion(args):
    if criterion_list.get(args.criterion, None) is not None:
        return criterion_list[args.criterion](args)
    else:
        raise NameError(f"Undefined criterion for {args.criterion}!")
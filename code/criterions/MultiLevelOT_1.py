import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .various_divergence import VariousDivergence
from .cross_entropy_loss import CrossEntropyLoss
import editdistance
import cvxpy as cp
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from .ETP_1 import ETP_1
from .ETP import ETP


def normalize(value):
    means = value.mean(dim=-1, keepdim=True)
    stds = value.std(dim=-1, keepdim=True)
    z_score_normalized_student = (value) / (stds + 0.0001)
    return z_score_normalized_student

def improved_sort(value):
    sums = value.sum(dim=(0, 1))
    sorted_indices = torch.argsort(sums, descending=True)
    sorted_values = value[:, :, sorted_indices]
    return sorted_values

def KL_wo(y_s, y_t, T=1):
    p_s = F.log_softmax(y_s / T, dim=-1)
    p_t = F.softmax(y_t / T, dim=-1)
    loss = -torch.sum(p_t * p_s, dim=-1).mean()
    return loss

class Sinkhorn_seq(nn.Module):
    def __init__(self, T=2):
        super(Sinkhorn_seq, self).__init__()
        self.T = T

    def sinkhorn_normalized(self, x, n_iters=20):
        for _ in range(n_iters):
            x = x / torch.sum(x, dim=1, keepdim=True)
            x = x / torch.sum(x, dim=0, keepdim=True)
        return x

    def sinkhorn_loss(self, x, y, epsilon=0.1, n_iters=10):
        Wxy = torch.cdist(x, y, p=1)
        K = torch.exp(-Wxy / epsilon)
        P = self.sinkhorn_normalized(K, n_iters)
        return torch.sum(P * Wxy)

    def forward(self, y_s, y_t):
        softmax = nn.Softmax(dim=-1)
        p_s = softmax(y_s / self.T)
        p_t = softmax(y_t / self.T)
        emd_loss = 0
        for i in range(p_s.shape[0]):
            emd_loss += 0.001 * self.sinkhorn_loss(x=p_s[i], y=p_t[i])
        return emd_loss

def greedy_algorithm_adjust_s(t, s):
    batch_size, T, k = t.shape
    _, n, _ = s.shape
    
    # Initialize the adjusted source tensor
    s_adjusted = torch.zeros_like(t)
    
    for b in range(batch_size):
        # Initialize set of available source indices for each batch
        available_indices = list(range(n))
        
        for i in range(T):
            C_min = float('inf')
            j_star = -1
            
            for j in available_indices:
                # Compute cost as the sum of absolute differences for each batch
                C = torch.sum(torch.abs(t[b, :, i] - s[b, :, j]))
                
                if C < C_min:
                    C_min = C
                    j_star = j
            
            # Assign the best matching source vector to the adjusted tensor
            s_adjusted[b, :, i] = s[b, :, j_star]
            
            # Remove the selected index from available indices
            available_indices.remove(j_star)

    return s_adjusted

class MultiLevelOT_1(VariousDivergence):
    def __init__(self, args, padding_id=-100) -> None:
        super().__init__(args, padding_id=padding_id)
        print(f"-------------Using MultiLevelOT + MultiCost-------------")
        self.args = args

        self.student_temperature = 2.0
        self.teacher_temperature = 2.0
        self.f = 1

        if torch.cuda.is_available() and args.precision == "bf16":
            self.dtype = torch.bfloat16
        elif torch.cuda.is_available() and args.precision == "fp16":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"ULD + MultiCost\n")

        self.window_size = 4
        self.padding_id = padding_id
        self.ot_weight_logits = args.ot_weight_logits
        self.ot_weight_hidden = args.ot_weight_hidden
        self.ce_ = args.ce_weight
        self.kd_rate = args.kd_rate
        self.tau_seq = 2.0
        self.top_k_vocab = args.top_k_vocab
        self.total_steps = args.total_iters
        self.current_step = 0
        self.sigma = 0.7
        self._id_mapping_cache = None

        d_s = args.hidden_dim_student
        d_t = args.hidden_dim_teacher 
        self.salience_proj_s = nn.Linear(d_s, 1, bias=True).to(self.device, dtype=self.dtype)

        self.etp = ETP()
        self.cost_weights_logits = nn.Parameter(torch.tensor([0.2, 0.5, 0.3], dtype=self.dtype, device=self.device))
        self.cost_weights_hidden = nn.Parameter(torch.tensor([0.3, 0.5, 0.2], dtype=self.dtype, device=self.device))

        print(f"ot_weight_logits: {self.ot_weight_logits}")
        print(f"ot_weight_hidden: {self.ot_weight_hidden}")
        print(f"kd_rate: {self.kd_rate}")
        print(f"ce_: {self.ce_}")
        print(f"top_k_vocab: {self.top_k_vocab}")



    def forward(
        self, 
        distiller, 
        input_data, 
        output_data, 
        logging_output, 
        batch_denom,
    ):
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        self.distiller = distiller

        # Add
        self.current_step += 1
        self.distiller.input_data = input_data

        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_data[f"teacher_{distiller.teacher_model_type}_input_ids"],
                attention_mask=input_data[f"teacher_{distiller.teacher_model_type}_attention_mask"],
                position_ids=input_data.get(f"teacher_{distiller.teacher_model_type}_position_ids", None), 
                output_hidden_states=True
            )

        outputs = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            position_ids=input_data.get("position_ids", None), 
            output_hidden_states=True
        )

        logits = outputs.logits
        log = {}
        loss_ce = self.compute_cross_entropy_loss(
            outputs.logits, output_data["label"], log=log
        )[0]

        pad_mask = input_data["attention_mask"].bool()
        teacher_pad_mask = input_data[f"teacher_{distiller.teacher_model_type}_attention_mask"].bool()

        ot_loss_logits, log = self.compute_ot_logits(distiller, outputs.logits, teacher_outputs.logits, 
                                        pad_mask, teacher_pad_mask, outputs.hidden_states[-1], teacher_outputs.hidden_states[-1], log)
        ot_loss_hidden, log = self.compute_ot_hidden(distiller, outputs.hidden_states[-1], teacher_outputs.hidden_states[-1], 
                                        pad_mask, teacher_pad_mask, log)
        
        kd_loss, log = self.compute_multi_level_ot_loss(
            outputs, teacher_outputs, output_data, distiller, log
        )
        log["kd_loss"] = kd_loss

        total_loss = self.ce_ * loss_ce + self.kd_rate * kd_loss + self.ot_weight_logits * ot_loss_logits + self.ot_weight_hidden * ot_loss_hidden
        log["loss"] = total_loss
        log["ot_loss_logits"] = ot_loss_logits
        log["ot_loss_hidden"] = ot_loss_hidden

        accuracy = self.compute_token_accuracy(
            logits, output_data["label"], 
        )
        log["accuracy"] = accuracy

        logging_output = self.record_logging_output(
            logging_output, batch_denom, log
        )
        return total_loss / batch_denom, logging_output


    def compute_multi_level_ot_loss(
        self, outputs, teacher_outputs, output_data, distiller, log
    ):
        student_target = output_data["label"]
        teacher_target = output_data[f"teacher_{distiller.teacher_model_type}_label"]
        student_logits = outputs.logits
        teacher_logits = teacher_outputs.logits

        # Chuẩn hóa logits
        student_logits = normalize(student_logits)
        teacher_logits = normalize(teacher_logits)

        # Căn chỉnh chuỗi student và teacher
        for i in range(student_target.shape[0]):
            stu_start_idx = student_target[i].ne(self.padding_id).nonzero()[0][0]
            tea_start_idx = teacher_target[i].ne(self.padding_id).nonzero()[0][0]
            student_target[i] = torch.cat([
                student_target[i][stu_start_idx:],
                student_target[i][:stu_start_idx]], dim=0
            )
            student_logits[i] = torch.cat([
                student_logits[i][stu_start_idx:, :],
                student_logits[i][:stu_start_idx, :]], dim=0
            )
            teacher_target[i] = torch.cat([
                teacher_target[i][tea_start_idx:],
                teacher_target[i][:tea_start_idx]], dim=0
            )
            teacher_logits[i] = torch.cat([
                teacher_logits[i][tea_start_idx:, :],
                teacher_logits[i][:tea_start_idx, :]], dim=0
            )

        # Áp dụng softmax với nhiệt độ
        student_probs = torch.softmax(student_logits / self.student_temperature, dim=-1, dtype=torch.float32)
        teacher_probs = torch.softmax(teacher_logits / self.teacher_temperature, dim=-1, dtype=torch.float32)

        # Sắp xếp xác suất theo thứ tự giảm dần
        student_probs = improved_sort(student_probs)
        teacher_probs = improved_sort(teacher_probs)

        # Cắt giảm kích thước từ điển (ví dụ: 50 token hàng đầu)
        top_k = 50
        student_probs = student_probs[:, :, :top_k]
        teacher_probs = teacher_probs[:, :, :top_k]

        # Điều chỉnh kích thước từ điển để khớp
        diff_size = student_probs.size(2) - teacher_probs.size(2)
        if diff_size > 0:
            teacher_probs = F.pad(teacher_probs, (0, diff_size), value=0)
        elif diff_size < 0:
            student_probs = F.pad(student_probs, (0, abs(diff_size)), value=0)

        # Áp dụng MultiLevel-OT
        sinkhorn = Sinkhorn_seq()
        if self.f == 1:
            # Sử dụng Sinkhorn trực tiếp
            ot_loss = sinkhorn(student_probs, teacher_probs) * 0.1
        elif self.f == 2:
            # Sử dụng greedy_algorithm_adjust_s để căn chỉnh
            adjusted_student_probs = greedy_algorithm_adjust_s(teacher_probs, student_probs)
            ot_loss = sinkhorn(adjusted_student_probs, teacher_probs)
        else:
            raise ValueError("Invalid value for f. Use 1 or 2.")

        # Thêm KL divergence
        kl_loss = KL_wo(student_probs, teacher_probs) * 0.1

        # Tổng hợp loss
        multi_loss = ot_loss + kl_loss + (student_probs - teacher_probs).abs().sum(-1).mean() 
        pad_mask = student_target.ne(self.padding_id) & teacher_target.ne(self.padding_id)
        multi_loss = (multi_loss * pad_mask).sum()

        log["loss"] = multi_loss
        return multi_loss, log
    

    def compute_ot_logits(self, distiller, student_logits, teacher_logits, student_mask, teacher_mask, student_outputs, teacher_outputs, log, t_start=0.1, t_end=1.0):
        batch_size = student_logits.size(0)
        tau = self.tau_seq
        eps = 1e-7
        k = self.top_k_vocab

        def normalize(value):
            means = value.mean(dim=-1, keepdim=True)
            stds = value.std(dim=-1, keepdim=True)
            return value / (stds + 0.0001)

        student_topk_logits, _ = student_logits.sort(dim=-1, descending=True)
        teacher_topk_logits, _ = teacher_logits.sort(dim=-1, descending=True)

        student_topk_logits = student_topk_logits[..., :k]
        teacher_topk_logits = teacher_topk_logits[..., :k]

        frac = min(self.current_step / self.total_steps, 1.0)
        t = t_start + (t_end - t_start) * frac
        interpolated_teacher_logits = (1 - t) * student_topk_logits + t * teacher_topk_logits

        student_probs = F.softmax(student_topk_logits / tau, dim=-1)
        interpolated_teacher_probs = F.softmax(interpolated_teacher_logits / tau, dim=-1)

        total_loss = 0
        for b in range(batch_size):
            mask_s = student_mask[b].bool()
            mask_t = teacher_mask[b].bool()
            sp = student_probs[b][mask_s]  # (N, k)
            tp = interpolated_teacher_probs[b][mask_t]  # (M, k)

            C2 = torch.cdist(tp, sp, p=2)  # (M, N)

            log_ratio = torch.log(tp.unsqueeze(1) / (sp.unsqueeze(0) + eps))  # (M, N, k)
            C4 = (tp.unsqueeze(1) * log_ratio).sum(dim=-1)  # (M, N)

            student_seq = student_outputs[b][mask_s]  # (N, hidden_dim)
            teacher_seq = distiller.projectors["ot"](teacher_outputs[b])[mask_t]  # (M, hidden_dim)
            sal_s = torch.sigmoid(self.salience_proj_s(student_seq.to(self.dtype))).squeeze(-1)
            sal_t = torch.sigmoid(self.salience_proj_s(teacher_seq.to(self.dtype))).squeeze(-1)  # (M,)
            C_salience = torch.abs(sal_t.unsqueeze(1) - sal_s.unsqueeze(0))  # (M, N)

            cost_matrices = [C2, C4, C_salience]
            for i, C in enumerate(cost_matrices):
                if C.shape != cost_matrices[0].shape:
                    raise ValueError(f"Cost matrix {i} has shape {C.shape}, expected {cost_matrices[0].shape}")
            weights = self.cost_weights_logits
            log["avg_c2_logits"] = C2.mean().item()
            log["avg_c4_logits"] = C4.mean().item()
            log["avg_c_salience_logits"] = C_salience.mean().item()

            total_cost = sum(w * C for w, C in zip(weights, cost_matrices))
            total_cost = total_cost.to(dtype=self.dtype)
            loss_etp, _ = self.etp(total_cost)
            total_loss += loss_etp

        loss = total_loss * self.ot_weight_logits / batch_size
        log["ot_loss_logits"] = loss.item()
        return loss, log

    def compute_ot_hidden(self, distiller, student_outputs, teacher_outputs, attention_mask_student, attention_mask_teacher, log):
        teacher_outputs = distiller.projectors["ot"](teacher_outputs)
        batch_size = teacher_outputs.size(0)
        total_loss = 0
        eps = 1e-7

        for b in range(batch_size):
            teacher_seq = teacher_outputs[b]
            student_seq = student_outputs[b]

            teacher_seq = teacher_seq[attention_mask_teacher[b].bool()]  # Shape: (valid_seq_len1, hidden_dim)
            student_seq = student_seq[attention_mask_student[b].bool()]  # Shape: (valid_seq_len2, hidden_dim)

            M = teacher_seq.size(0)  
            N = student_seq.size(0)  

            # C2
            student_ids = self.distiller.input_data["input_ids"][b][attention_mask_student[b].bool()].tolist()
            teacher_ids = self.distiller.input_data[f"teacher_{distiller.teacher_model_type}_input_ids"][b][attention_mask_teacher[b].bool()].tolist()
            stu_tok = distiller.student_tokenizer.convert_ids_to_tokens(student_ids, skip_special_tokens=True)
            tea_tok = distiller.teacher_tokenizers[distiller.teacher_model_type].convert_ids_to_tokens(teacher_ids, skip_special_tokens=True)

            edit_distance_cache = {}
            def safe_edit(a, b):
                key = (a, b)
                if key in edit_distance_cache:
                    return edit_distance_cache[key]
                val = editdistance.eval(a, b)
                edit_distance_cache[key] = val
                return val
            
            C_s2t = torch.zeros((N, M), device=student_seq.device)
            pairs_s2t = dtw_alignment(stu_tok, tea_tok, dist_fn_edit)  # student -> teacher
            for i, j in pairs_s2t:
                if i < N and j < M:
                    C_s2t[i, j] = safe_edit(stu_tok[i], tea_tok[j])

            C_t2s = torch.zeros((M, N), device=student_seq.device)
            pairs_t2s = dtw_alignment(tea_tok, stu_tok, dist_fn_edit)  # teacher -> student
            for j, i in pairs_t2s:
                if j < M and i < N:
                    C_t2s[j, i] = safe_edit(tea_tok[j], stu_tok[i])

            C2 = (C_s2t.T + C_t2s) / 2

            # C_contextual
            def compute_context_reprs(seq, window):
                ctx = torch.zeros_like(seq)
                for i in range(seq.size(0)):
                    start = max(i - window, 0)
                    end = min(i + window + 1, seq.size(0))
                    ctx[i] = seq[start:end].mean(dim=0)
                
                # add
                return ctx.to(self.dtype)
                # return ctx
            
            ctx_s = compute_context_reprs(student_seq, self.window_size)  
            ctx_t = compute_context_reprs(teacher_seq, self.window_size)
            # C5 = torch.cdist(ctx_t, ctx_s, p=2)
            # add
            C5 = torch.cdist(ctx_t.to(self.dtype), ctx_s.to(self.dtype), p=2)
            C5 = C5 / (C5.max() + eps)

            ctx_s_norm = ctx_s / (torch.norm(ctx_s, dim=-1, keepdim=True) + eps)
            ctx_t_norm = ctx_t / (torch.norm(ctx_t, dim=-1, keepdim=True) + eps)
            # cosine_sim = torch.einsum('md,nd->mn', ctx_t_norm, ctx_s_norm)
            # add
            cosine_sim = torch.einsum('md,nd->mn', ctx_t_norm.to(self.dtype), ctx_s_norm.to(self.dtype))
            C6 = 1 - cosine_sim 
            cost_matrices = [C2, C5, C6]
            for i, C in enumerate(cost_matrices):
                if C.shape != cost_matrices[0].shape:
                    raise ValueError(f"Cost matrix {i} has shape {C.shape}, expected {cost_matrices[0].shape}")
            weights = self.cost_weights_hidden
            log["avg_c2_last"] = C2.mean().item()
            log["avg_c5_last"] = C5.mean().item()
            log["avg_c6_last"] = C6.mean().item()
            # log["avg_c7_last"] = C7.mean().item()

            total_cost = sum(w * C for w, C in zip(weights, cost_matrices))
            total_cost = total_cost.to(dtype=self.dtype)
            loss_etp, _ = self.etp(total_cost)
            total_loss += loss_etp

        loss = total_loss * self.ot_weight_hidden
        loss = total_loss / batch_size
        log["ot_loss_hidden"] = loss.item()
        return loss, log

    
    def update_cost_weights(self, cost_values_logits, cost_values_hidden):
        def to_scalar_list(values):
            if isinstance(values, torch.Tensor):
                values = values.tolist()
            if not isinstance(values, list):
                logger.error(f"Expected list, got {type(values)}: {values}")
                return None
            try:
                result = []
                for x in values:
                    if isinstance(x, (int, float)):
                        result.append(float(x))
                    elif isinstance(x, (list, tuple)):
                        result.append(float(np.mean(x)))
                    else:
                        logger.error(f"Invalid value in list: {type(x)}, value: {x}")
                        return None
                return result
            except (TypeError, ValueError) as e:
                logger.error(f"Error converting to float: {e}, values: {values}")
                return None
        
        cost_values_logits = to_scalar_list(cost_values_logits)
        cost_values_logits = torch.tensor(cost_values_logits, dtype=self.dtype, device=self.device)

        cost_values_hidden = to_scalar_list(cost_values_hidden)
        cost_values_hidden = torch.tensor(cost_values_hidden, dtype=self.dtype, device=self.device)
        
        ###
        c_vals_logits = cost_values_logits.detach().cpu().float().numpy()
        n_logits = len(c_vals_logits)  
        sigma = self.sigma
        
        alpha_logits = cp.Variable(n_logits)
        objective_logits = cp.Minimize(c_vals_logits @ alpha_logits + sigma * cp.sum_squares(alpha_logits - 1/n_logits))
        constraints_logits = [cp.sum(alpha_logits) == 1, alpha_logits >= 0.01]
        
        problem_logits = cp.Problem(objective_logits, constraints_logits)
        problem_logits.solve(solver=cp.ECOS, verbose=False)
        
        if alpha_logits.value is None:
            print(f"Rank {dist.get_rank()}: CVXPY solver failed for cost_weights_logits. Skipping update.")
        else:
            new_weights_logits = torch.tensor(alpha_logits.value, dtype=self.cost_weights_logits.dtype, device=self.cost_weights_logits.device)
            with torch.no_grad():
                self.cost_weights_logits.copy_(new_weights_logits)
            alpha_str_logits = ", ".join([f"{w:.6f}" for w in new_weights_logits.tolist()])
            print(alpha_str_logits)
        
        ###
        c_vals_hidden = cost_values_hidden.detach().cpu().float().numpy()
        n_hidden = len(c_vals_hidden) 
        sigma = self.sigma
        
        alpha_hidden = cp.Variable(n_hidden)
        objective_hidden = cp.Minimize(c_vals_hidden @ alpha_hidden + sigma * cp.sum_squares(alpha_hidden - 1/n_hidden))
        constraints_hidden = [cp.sum(alpha_hidden) == 1, alpha_hidden >= 0.01]
        
        problem_hidden = cp.Problem(objective_hidden, constraints_hidden)
        problem_hidden.solve(solver=cp.ECOS, verbose=False)
        
        if alpha_hidden.value is None:
            print(f"Rank {dist.get_rank()}: CVXPY solver failed for cost_weights_hidden. Skipping update.")
        else:
            new_weights_hidden = torch.tensor(alpha_hidden.value, dtype=self.cost_weights_hidden.dtype, device=self.cost_weights_hidden.device)
            with torch.no_grad():
                self.cost_weights_hidden.copy_(new_weights_hidden)
            alpha_str_hidden = ", ".join([f"{w:.6f}" for w in new_weights_hidden.tolist()])
            print(alpha_str_hidden)





def dist_fn_edit(a, b):
    return editdistance.eval(a, b)

def dtw_alignment(series_1, series_2, norm_func=dist_fn_edit):
    """Simple DTW based on FUSELLM"""
    matrix = np.zeros((len(series_1) + 1, len(series_2) + 1))
    matrix[0, :] = np.inf
    matrix[:, 0] = np.inf
    matrix[0, 0] = 0
    for i, vec1 in enumerate(series_1):
        for j, vec2 in enumerate(series_2):
            cost = norm_func(vec1, vec2)
            matrix[i + 1, j + 1] = cost + min(
                matrix[i, j + 1], matrix[i + 1, j], matrix[i, j]
            )
    matrix = matrix[1:, 1:]
    i, j = len(series_1) - 1, len(series_2) - 1
    aligned = []
    while i > 0 or j > 0:
        aligned.append((i, j))
        options = [
            matrix[i - 1, j - 1] if i > 0 and j > 0 else np.inf,
            matrix[i - 1, j] if i > 0 else np.inf,
            matrix[i, j - 1] if j > 0 else np.inf,
        ]
        move = np.argmin(options)
        if move == 0: i -= 1; j -= 1
        elif move == 1: i -= 1
        else: j -= 1
    aligned.append((0, 0))
    return aligned


def pairwise_euclidean_distance(x, y):
    return torch.cdist(x, y, p=2) 

def pairwise_cosin_distance(a, b, eps=1e-8):
    # a = a.float()
    # b = b.float()
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n, dtype=torch.bfloat16))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n, dtype=torch.bfloat16))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    sim_mt = 1 - sim_mt
    return sim_mt

def pairwise_attention_distance(x, y, eps=1e-8):
    # x = x.float()
    # y = y.float()
    d = x.shape[1]
    sim_mt = torch.mm(x, y.transpose(0, 1)) / math.sqrt(d)
    attention_weights = torch.softmax(sim_mt, dim=1)

    dist_mt = 1.0 - attention_weights
    return dist_mt
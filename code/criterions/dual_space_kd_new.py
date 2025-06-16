import math
import torch
from .various_divergence import VariousDivergence
from .ETP_1 import ETP_1
from .ETP import ETP
import editdistance
import cvxpy as cp
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist


class DualSpaceKDWithCMA_OT(VariousDivergence):
    def __init__(self, args, padding_id=-100):
        super().__init__(args, padding_id=padding_id)
        print("--------------------Using KB Su dung Multi-OT-New-------------------")
        self.args = args
        
        if torch.cuda.is_available() and args.precision == "bf16":
            self.dtype = torch.bfloat16
        elif torch.cuda.is_available() and args.precision == "fp16":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ## Add
        # Tot nhat la 2.5, 10.0, 1.7, 300, 100.0, 100.0, 0.7 -> dc 19.5. 1000 thi te hon
        # 2.5, 10.0, 1.7, 150, 100.0, 100.0, 0.7
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
        self.current_step += 1
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        self.distiller = distiller
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

        loss_ce = self.compute_cross_entropy_loss(outputs.logits, output_data["label"], log=log)[0]
        log["loss_ce"] = loss_ce

        hidden_state_student = outputs.hidden_states[-1]  # (batch_size, seq_len_student, hidden_dim_student)
        # hidden_state_student_first = outputs.hidden_states[0]
        hidden_state_teacher = teacher_outputs.hidden_states[-1]  # (batch_size, seq_len_teacher, hidden_dim_teacher)
        # hidden_state_teacher_first = teacher_outputs.hidden_states[0]
        
        pad_mask = input_data["attention_mask"].bool()
        teacher_pad_mask = input_data[f"teacher_{distiller.teacher_model_type}_attention_mask"].bool()

        ot_loss_logits, log = self.compute_ot_logits(distiller, outputs.logits, teacher_outputs.logits, 
                                        pad_mask, teacher_pad_mask, outputs.hidden_states[-1], teacher_outputs.hidden_states[-1], log)
        ot_loss_hidden, log = self.compute_ot_hidden(distiller, outputs.hidden_states[-1], teacher_outputs.hidden_states[-1], 
                                        pad_mask, teacher_pad_mask, log)
        
        kd_loss, log = self.compute_dual_space_kd_loss_with_cma(
            outputs, teacher_outputs, input_data, output_data, distiller, log
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
        

    def compute_ot_logits(self, distiller, student_logits, teacher_logits, student_mask, teacher_mask, student_outputs, teacher_outputs, log, t_start=0.1, t_end=1.0):
        batch_size = student_logits.size(0)
        tau = self.tau_seq
        eps = 1e-7
        k = self.top_k_vocab

        def normalize(value):
            means = value.mean(dim=-1, keepdim=True)
            stds = value.std(dim=-1, keepdim=True)
            return value / (stds + 0.0001)

        # student_logits = normalize(student_logits).to(self.dtype)
        # teacher_logits = normalize(teacher_logits).to(self.dtype)

        # student_probs = F.softmax(student_logits / tau, dim=-1)
        # teacher_probs = F.softmax(teacher_logits / tau, dim=-1)

        # min_vocab = min(student_probs.size(-1), teacher_probs.size(-1))
        # k = min(min_vocab, self.top_k_vocab)

        # student_prob_sums = student_probs.sum(dim=(0, 1))
        # teacher_prob_sums = teacher_probs.sum(dim=(0, 1))

        # _, student_topk_indices = torch.topk(student_prob_sums, k=k, dim=-1)
        # _, teacher_topk_indices = torch.topk(teacher_prob_sums[:min_vocab], k=k, dim=-1)

        # selected_indices = student_topk_indices

        # student_logits = student_logits[:, :, selected_indices]
        # teacher_logits = teacher_logits[:, :, selected_indices]

        # frac = self.current_step / self.total_steps
        # frac = min(frac, 1.0)               
        # t = t_start + (t_end - t_start) * frac
        # # # t = min(self.current_step / self.total_steps, 1.0)
        # # t = 0.5 * (1 - math.cos(math.pi * self.current_step / self.total_steps))
        # interpolated_teacher_logits = (1 - t) * student_logits + t * teacher_logits

        # student_probs = F.softmax(student_logits / tau, dim=-1)
        # interpolated_teacher_probs = F.softmax(interpolated_teacher_logits / tau, dim=-1)

        # def improved_sort(value):
        #     sums = value.sum(dim=(0, 1))
        #     sorted_indices = torch.argsort(sums, descending=True)
        #     return value[:, :, sorted_indices]

        # student_probs = improved_sort(student_probs)
        # interpolated_teacher_probs = improved_sort(interpolated_teacher_probs)
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

            # # C_hybrid
            # def compute_ngram_overlap_cost(stu_tok, tea_tok, student_ids, teacher_ids, n=2):
            #     stu_text = distiller.student_tokenizer.decode(student_ids, skip_special_tokens=True).lower()
            #     tea_text = distiller.teacher_tokenizers[distiller.teacher_model_type].decode(teacher_ids, skip_special_tokens=True).lower()
                
            #     word_tokens_stu = stu_text.split()
            #     word_tokens_tea = tea_text.split()
                
            #     stu_ngrams = set(tuple(word_tokens_stu[i:i+n]) for i in range(len(word_tokens_stu)-n+1))
            #     tea_ngrams = set(tuple(word_tokens_tea[i:i+n]) for i in range(len(word_tokens_tea)-n+1))
            #     common_ngrams = stu_ngrams & tea_ngrams
                
            #     if self._id_mapping_cache is None:
            #         tea2stu_id_mapping = {}
            #         # Use tea2stu_token_mapping if available
            #         # if hasattr(distiller, 'tea2stu_token_mapping'):
            #         #     for t_tok, s_tok in distiller.tea2stu_token_mapping.items():
            #         #         t_id = distiller.teacher_tokenizers[distiller.teacher_model_type].convert_tokens_to_ids(t_tok)
            #         #         s_id = distiller.student_tokenizer.convert_tokens_to_ids(s_tok)
            #         #         if t_id is not None and s_id is not None:
            #         #             tea2stu_id_mapping[str(t_id)] = s_id
            #         # Add direct token-to-id mapping
            #         for t_id in set(teacher_ids):
            #             t_tok = distiller.teacher_tokenizers[distiller.teacher_model_type].convert_ids_to_tokens(t_id)
            #             s_id = distiller.student_tokenizer.convert_tokens_to_ids(t_tok)
            #             if s_id is not None:
            #                 tea2stu_id_mapping[str(t_id)] = s_id
            #         self._id_mapping_cache = tea2stu_id_mapping
                
            #     # Use cached mapping
            #     C_ngram = torch.ones((N, M), device=student_seq.device)
            #     for i, s_id in enumerate(student_ids):
            #         for j, t_id in enumerate(teacher_ids):
            #             t_id_str = str(t_id)
            #             if t_id_str in self._id_mapping_cache and self._id_mapping_cache[t_id_str] == s_id:
            #                 # Check n-gram overlap for aligned tokens
            #                 stu_text_pos = stu_text
            #                 tea_text_pos = tea_text
            #                 for ngram in common_ngrams:
            #                     ngram_str = ' '.join(ngram)
            #                     if ngram_str in stu_text_pos and ngram_str in tea_text_pos:
            #                         C_ngram[i, j] = 0
            #                         stu_text_pos = stu_text_pos.replace(ngram_str, '', 1)
            #                         tea_text_pos = tea_text_pos.replace(ngram_str, '', 1)
            #                         break
            #     # Fallback to word-level mapping if no mapping
            #     if not self._id_mapping_cache:
            #         stu_word_map = {}
            #         tea_word_map = {}
            #         word_idx = 0
            #         stu_idx = 0
            #         tea_idx = 0
            #         stu_text_remaining = stu_text.replace('##', '')
            #         tea_text_remaining = tea_text.replace('##', '')
                    
            #         for word in word_tokens_stu:
            #             while stu_idx < len(stu_tok) and word in stu_text_remaining:
            #                 stu_word_map[stu_idx] = word_idx
            #                 stu_text_remaining = stu_text_remaining.replace(word, '', 1)
            #                 stu_idx += 1
            #             word_idx += 1
            #         word_idx = 0
            #         for word in word_tokens_tea:
            #             while tea_idx < len(tea_tok) and word in tea_text_remaining:
            #                 tea_word_map[tea_idx] = word_idx
            #                 tea_text_remaining = tea_text_remaining.replace(word, '', 1)
            #                 tea_idx += 1
            #             word_idx += 1

            #         for i in range(N):
            #             for j in range(M):
            #                 if i in stu_word_map and j in tea_word_map:
            #                     stu_word_idx = stu_word_map[i]
            #                     tea_word_idx = tea_word_map[j]
            #                     if stu_word_idx < len(word_tokens_stu) and tea_word_idx < len(word_tokens_tea):
            #                         for ngram in common_ngrams:
            #                             if (word_tokens_stu[stu_word_idx] in ngram and
            #                                 word_tokens_tea[tea_word_idx] in ngram):
            #                                 C_ngram[i, j] = 0
            #                                 break
            #     C_ngram = C_ngram / (C_ngram.max() + eps)
            #     return C_ngram.T
            
            # C7 = compute_ngram_overlap_cost(stu_tok, tea_tok, student_ids, teacher_ids, n=2)

            # cost_matrices = [C2, C5, C6, C7]
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



    def compute_dual_space_kd_loss_with_cma(
        self, outputs, teacher_outputs, input_data, output_data, distiller, log
    ):
        # Ground truth labels
        target = output_data["label"]

        teacher_target = output_data[f"teacher_{distiller.teacher_model_type}_label"]
        pad_mask = target.ne(self.padding_id)
        teacher_pad_mask = teacher_target.ne(self.padding_id)

        # hiddens = outputs.hidden_states[-1]
        # teacher_hiddens = teacher_outputs.hidden_states[-1]
        hiddens = outputs.hidden_states[-1].to(self.dtype)
        teacher_hiddens = teacher_outputs.hidden_states[-1].to(self.dtype)

        if hasattr(distiller.student_model, "model") \
            and hasattr(distiller.student_model.model, "embed_tokens"):
            stu_embed_tokens = distiller.student_model.model.embed_tokens
        elif hasattr(distiller.student_model, "model") \
            and hasattr(distiller.student_model.model, "model") \
            and hasattr(distiller.student_model.model.model, "embed_tokens"):
            stu_embed_tokens = distiller.student_model.model.model.embed_tokens
        elif hasattr(distiller.student_model, "transformer") \
            and hasattr(distiller.student_model.transformer, "wte"):
            stu_embed_tokens = distiller.student_model.transformer.wte
        else:
            raise NotImplementedError

        if hasattr(distiller.teacher_model, "model") \
            and hasattr(distiller.teacher_model.model, "embed_tokens"):
            tea_embed_tokens = distiller.teacher_model.model.embed_tokens
        elif hasattr(distiller.teacher_model, "model") \
            and hasattr(distiller.teacher_model.model, "model") \
            and hasattr(distiller.teacher_model.model.model, "embed_tokens"):
            tea_embed_tokens = distiller.teacher_model.model.model.embed_tokens
        elif hasattr(distiller.teacher_model, "transformer") \
            and hasattr(distiller.teacher_model.model, "wte"):
            tea_embed_tokens = distiller.teacher_model.transformer.wte
        else:
            raise NotImplementedError

        formal_target = torch.where(pad_mask, target, torch.zeros_like(target))
        formal_input = torch.where(pad_mask, input_data["input_ids"], torch.zeros_like(target))

        # Add
        # stu_input_embeds = stu_embed_tokens(formal_input).detach()
        # stu_target_embeds = stu_embed_tokens(formal_target).detach()
        stu_input_embeds = stu_embed_tokens(formal_input).detach().to(self.dtype)
        stu_target_embeds = stu_embed_tokens(formal_target).detach().to(self.dtype)

        formal_teacher_target = torch.where(teacher_pad_mask, teacher_target, torch.zeros_like(teacher_target))
        formal_teacher_input = torch.where(teacher_pad_mask, input_data[f"teacher_{distiller.teacher_model_type}_input_ids"], torch.zeros_like(teacher_target))
        # tea_input_embeds = tea_embed_tokens(formal_teacher_input).detach()
        # tea_target_embeds = tea_embed_tokens(formal_teacher_target).detach()
        tea_input_embeds = tea_embed_tokens(formal_teacher_input).detach().to(self.dtype)
        tea_target_embeds = tea_embed_tokens(formal_teacher_target).detach().to(self.dtype)

        # stu_index_embeds = torch.cat([stu_input_embeds, stu_target_embeds], -1)
        # tea_index_embeds = torch.cat([tea_input_embeds, tea_target_embeds], -1)
        stu_index_embeds = torch.cat([stu_input_embeds, stu_target_embeds], -1).to(self.dtype)
        tea_index_embeds = torch.cat([tea_input_embeds, tea_target_embeds], -1).to(self.dtype)

        norm_tea_index_embeds = tea_index_embeds / tea_index_embeds.std()
        norm_tea_target_embeds = tea_target_embeds / tea_target_embeds.std()
        norm_teacher_hiddens = teacher_hiddens / teacher_hiddens.std()

        ## Add
        # print(f"stu_index_embeds shape: {stu_index_embeds.shape}, dtype: {stu_index_embeds.dtype}")
        # print(f"tea_index_embeds shape: {tea_index_embeds.shape}, dtype: {tea_index_embeds.dtype}")
        # print(f"hiddens shape: {hiddens.shape}, dtype: {hiddens.dtype}")
        # print(f"norm_teacher_hiddens shape: {norm_teacher_hiddens.shape}, dtype: {norm_teacher_hiddens.dtype}")

        stu_q_hiddens = distiller.projectors["query"](stu_index_embeds.to(self.dtype))
        tea_k_hiddens = norm_tea_index_embeds.to(self.dtype)

        stu_v_hiddens = distiller.projectors["s2t"](hiddens.to(self.dtype))
        tea_v_hiddens = distiller.projectors["t2s"](
            (norm_teacher_hiddens + norm_tea_target_embeds).to(self.dtype)
        )

        # align = stu_q_hiddens.matmul(tea_k_hiddens.transpose(-1, -2))
        align = stu_q_hiddens.matmul(tea_k_hiddens.transpose(-1, -2)).to(self.dtype)
        align = align / math.sqrt(2 * teacher_hiddens.shape[-1])
        align_mask = pad_mask.float().unsqueeze(-1) * teacher_pad_mask.float().unsqueeze(1)
        align = align + (1.0 - align_mask) * (-100000)

        # t2s_weight = torch.softmax(align, -1)        
        # t2s_hiddens = t2s_weight.matmul(tea_v_hiddens).to(hiddens)
        t2s_weight = torch.softmax(align, -1).to(self.dtype)
        # print(f"t2s_weight shape: {t2s_weight.shape}, dtype: {t2s_weight.dtype}")
        t2s_hiddens = t2s_weight.matmul(tea_v_hiddens).to(hiddens)
        t2s_logits = t2s_hiddens.matmul(
            distiller.student_model.lm_head.weight.detach().transpose(-1, -2).to(self.dtype)
        )
        # t2s_logits = t2s_hiddens.matmul(
        #     distiller.student_model.lm_head.weight.detach().transpose(-1, -2)
        # )

        t2s_ce_loss = self.compute_cross_entropy_loss(t2s_logits, target)[0]
        t2s_acc_mask = t2s_logits.argmax(-1).eq(target)
        t2s_acc = (t2s_acc_mask * pad_mask).sum()
        max_probs = (t2s_logits.softmax(-1).max(-1)[0] * pad_mask).sum()
        log["t2s_ce_loss"] = t2s_ce_loss
        log["t2s_acc"] = t2s_acc
        log["max_t2s_prob"] = max_probs
        
        if not self.args.only_save_projector:  # skip if only train projectors (pre-train projectors)
            t2s_kd_loss = self.dist_func(
                outputs.logits, t2s_logits.detach(), target, reduction="none", use_tea_temp=True
            )
            t2s_kd_loss = (t2s_kd_loss * pad_mask * t2s_acc_mask).sum()

            # s2t_weight = torch.softmax(align.transpose(-1, -2), -1)
            # s2t_hiddens = s2t_weight.matmul(stu_v_hiddens).to(hiddens)
            # s2t_logits = s2t_hiddens.matmul(
            # distiller.teacher_model.lm_head.weight.detach().transpose(-1, -2)
            # )
            s2t_weight = torch.softmax(align.transpose(-1, -2), -1).to(self.dtype)
            s2t_hiddens = s2t_weight.matmul(stu_v_hiddens).to(self.dtype)
            s2t_logits = s2t_hiddens.matmul(
                distiller.teacher_model.lm_head.weight.detach().transpose(-1, -2).to(self.dtype)
            )
            s2t_kd_loss = self.compute_forward_kl_divergence(
                s2t_logits, teacher_outputs.logits.to(self.dtype), teacher_target, reduction="none"
            )
            # s2t_kd_loss = self.compute_forward_kl_divergence(
            #     s2t_logits, teacher_outputs.logits, teacher_target, reduction="none"
            # )
            s2t_kd_loss = (s2t_kd_loss * teacher_pad_mask).sum()
            s2t_acc = (s2t_logits.argmax(-1).eq(teacher_target) * teacher_pad_mask).sum() * pad_mask.sum() / teacher_pad_mask.sum()

            kd_loss = t2s_ce_loss + t2s_kd_loss + s2t_kd_loss
            # kd_loss = t2s_kd_loss + s2t_kd_loss
            log["t2s_kd_loss"] = t2s_kd_loss
            log["s2t_kd_loss"] = s2t_kd_loss
            log["s2t_acc"] = s2t_acc

        log["kd_loss"] = kd_loss
        return kd_loss, log



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



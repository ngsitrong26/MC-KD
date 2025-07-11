﻿import logging
import torch
import torch.distributed
import torch.nn.functional as F
import numpy as np
import transformers
import editdistance
from typing import Dict, List
from .various_divergence import VariousDivergence
import math
from .ETP_1 import ETP_1
from .ETP import ETP
import cvxpy as cp
import torch.nn as nn
import torch.distributed as dist



TOKENIZER_TO_SPECIAL_TOKEN = {
    transformers.LlamaTokenizer: "▁",
    transformers.LlamaTokenizerFast: "▁",
    transformers.GPTNeoXTokenizerFast: "Ġ",
    transformers.GPT2Tokenizer: "Ġ",
    transformers.GPT2TokenizerFast: "Ġ",
    transformers.Qwen2Tokenizer: "Ġ",
    transformers.Qwen2TokenizerFast: "Ġ",
}

class MinEditDisForwardKLD_1(VariousDivergence):
    def __init__(self, args, padding_id=-100) -> None:
        super(MinEditDisForwardKLD_1, self).__init__(args, padding_id=padding_id)
        self.kd_rate = args.kd_rate
        self.kd_temp = args.kd_temperature

        print(f"MinED + MultiCost")
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
                output_hidden_states=True)

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
            
        teacher_logits = self.get_aligned_teacher_logits(
            logits, 
            teacher_outputs.logits, 
            input_data,
            output_data,
            distiller
        )
        
        kd_loss = self.compute_forward_kl_divergence(
            logits, 
            teacher_logits, 
            output_data["label"],
            log=log
        )

        total_loss = self.ce_ * loss_ce + self.kd_rate * kd_loss + self.ot_weight_logits * ot_loss_logits + self.ot_weight_hidden * ot_loss_hidden
        log["loss"] = total_loss
        log["ot_loss_logits"] = ot_loss_logits
        log["ot_loss_hidden"] = ot_loss_hidden

        accuracy = self.compute_token_accuracy(
            logits, 
            output_data["label"], 
        )
        log["accuracy"] = accuracy

        logging_output = self.record_logging_output(
            logging_output, 
            batch_denom,
            log
        )
        return total_loss / batch_denom, logging_output

    def get_aligned_teacher_logits(
        self, logits, teacher_logits, input_data, output_data, distiller,
    ):
        target = output_data["label"]
        pad_mask = target.ne(self.padding_id)
        teacher_target = output_data[f"teacher_{distiller.teacher_model_type}_label"]
        target_ids = torch.where(
            pad_mask, 
            target, 
            torch.ones_like(target) * distiller.student_tokenizer.eos_token_id
        )
        stu_tokenizer = distiller.student_tokenizer
        tea_tokenizer = distiller.teacher_tokenizers[distiller.teacher_model_type]

        bsz = target.shape[0]
        aligned_tea_logits = []
        for i in range(bsz):
            stu_content_idx = torch.nonzero(target[i].ne(self.padding_id)).view(-1)
            stu_input_ids = input_data["input_ids"][i, stu_content_idx]
            stu_target_ids = target_ids[i, stu_content_idx]

            tea_content_idx = torch.nonzero(teacher_target[i].ne(self.padding_id)).view(-1)
            tea_input_ids = input_data[f"teacher_{distiller.teacher_model_type}_input_ids"][i, tea_content_idx]

            stu_per_step_logits = logits[i, stu_content_idx, :].float()
            tea_per_step_logits = teacher_logits[i, tea_content_idx, :].float()   # [slen, vocab]

            aligned_tea_content_per_step_logits = self.transform_step_logits_fast(
                stu_tokenizer,
                tea_tokenizer,
                stu_input_ids,
                stu_per_step_logits,
                stu_target_ids,
                tea_input_ids,
                tea_per_step_logits,
                blending_to_base_mapping=distiller.tea2stu_id_mapping,
                base_to_blending_mapping_blending_ids=distiller.stu2tea_id_mapping_tea,
                base_to_blending_mapping_base_ids=distiller.stu2tea_id_mapping_stu
            )

            aligned_tea_per_step_logits = logits[i].float().detach()
            aligned_tea_per_step_logits[stu_content_idx] = aligned_tea_content_per_step_logits
            aligned_tea_logits.append(aligned_tea_per_step_logits)
        
        aligned_tea_logits = torch.stack(aligned_tea_logits, 0)
        return aligned_tea_logits

    def transform_step_logits_fast(
        self,
        base_model_tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase,
        blending_model_tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase,
        base_model_input_ids: torch.LongTensor,
        base_model_per_step_logits: torch.FloatTensor,
        base_model_target_ids: torch.LongTensor,
        blending_model_input_ids: torch.LongTensor,
        blending_model_per_step_logits: torch.FloatTensor,
        blending_to_base_mapping: torch.LongTensor = None,
        base_to_blending_mapping_blending_ids: torch.LongTensor = None,
        base_to_blending_mapping_base_ids: torch.LongTensor = None,
        device: str = None,
    ):
        """faster implementation to align logits"""
        base_model_tokens = base_model_tokenizer.convert_ids_to_tokens(base_model_input_ids)
        blending_model_tokens = blending_model_tokenizer.convert_ids_to_tokens(
            blending_model_input_ids
        )
        base_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[
            base_model_tokenizer.__class__
        ]
        blending_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[
            blending_model_tokenizer.__class__
        ]

        def dist_fn(a, b):
            """Calculate editdistance between two tokens, a is from blending model, b is from base model."""
            aa = a.replace(blending_model_special_token, "")
            bb = b.replace(base_model_special_token, "")
            dist = editdistance.eval(aa, bb)
            return dist

        # obtain sequence token alignment (each stu token to which tea token)
        _, _, _, base_to_blending, _ = self.dtw(
            blending_model_tokens, base_model_tokens, norm_func=dist_fn
        ) 
        unalign_mask = [1 if len(a) == 1 else 0 for a in base_to_blending]
        unalign_mask = torch.tensor(unalign_mask).to(base_model_input_ids.device)

        # for one-to-one mapping, align their logits; for one-to-many mapping, use ground-truth one-hot target
        base_to_blending = [a[0] if len(a) == 1 else 0 for a in base_to_blending]
        base_to_blending = torch.LongTensor(base_to_blending).to(base_model_input_ids.device)
        # for one-to-one mapping, ensure they are really similar
        unalign_mask = unalign_mask & base_model_input_ids.eq(blending_to_base_mapping[blending_model_input_ids[base_to_blending]])
        # get the logits of mapped tea tokens
        blending_model_per_step_logits = blending_model_per_step_logits[base_to_blending]
        blending_model_per_step_logits = blending_model_per_step_logits[
            :, base_to_blending_mapping_blending_ids.view(-1)
        ]
        blending_model_per_step_logits = blending_model_per_step_logits.view(
            -1, 
            base_to_blending_mapping_blending_ids.shape[0], 
            base_to_blending_mapping_blending_ids.shape[1]
        ).max(-1)[0]
        # transform teacher logits to student logits
        blending_to_base_logits = torch.ones_like(base_model_per_step_logits) * (-100000)
        blending_to_base_logits[:, base_to_blending_mapping_base_ids] = blending_model_per_step_logits
        
        unalign_mask = unalign_mask \
                     & blending_to_base_logits.max(-1)[0].ne(-100000)
        # mask unaligned position, use ground-truth target (one-hot)
        one_hot_logits = F.one_hot(base_model_target_ids, num_classes=base_model_per_step_logits.shape[-1])
        one_hot_logits = (1 - one_hot_logits) * (-100000) + (one_hot_logits) * 100
        
        unalign_mask = unalign_mask.unsqueeze(-1)
        blending_to_base_logits = torch.where(
            unalign_mask.repeat(1, base_model_per_step_logits.shape[-1]).eq(1),
            blending_to_base_logits,
            one_hot_logits
        )

        return blending_to_base_logits


    def transform_step_logits(
        self,
        base_model_tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase,
        blending_model_tokenizer: transformers.tokenization_utils_base.PreTrainedTokenizerBase,
        base_model_vocab: Dict[str, int],
        base_model_input_ids: List[int],
        blending_model_input_ids: List[int],
        blending_model_per_step_logits: List[List[float]],
        blending_model_per_step_indices: List[List[int]],
        vocab_align_type: str = "hard",
        blending_to_base_mapping: Dict[str, str] = None,
    ):
        """Align blending model per step logits & indices with base model. (original implementation in FuseLLM)"""
        base_model_tokens = base_model_tokenizer.convert_ids_to_tokens(base_model_input_ids)
        blending_model_tokens = blending_model_tokenizer.convert_ids_to_tokens(
            blending_model_input_ids
        )
        base_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[
            base_model_tokenizer.__class__
        ]
        blending_model_special_token = TOKENIZER_TO_SPECIAL_TOKEN[
            blending_model_tokenizer.__class__
        ]

        def dist_fn(a, b):
            """Calculate editdistance between two tokens, a is from blending model, b is from base model."""
            aa = a.replace(blending_model_special_token, "")
            bb = b.replace(base_model_special_token, "")
            dist = editdistance.eval(aa, bb)
            return dist

        _, _, _, base_to_blending, _ = self.dtw(
            blending_model_tokens, base_model_tokens, norm_func=dist_fn
        )
        aligned_blending_model_per_step_logits, aligned_blending_model_per_step_indices = (
            [],
            [],
        )
        for i, blending_idx in enumerate(base_to_blending):
            aligned_blending_model_per_step_logit = []
            aligned_blending_model_per_step_index = []
            if len(blending_idx) == 1:  # one base token map to one blending token
                j = blending_idx[0]
                base_token = base_model_tokens[i]
                blending_token = blending_model_tokens[j].replace(
                    blending_model_special_token, base_model_special_token
                )
                if (
                    (
                        blending_model_tokenizer.__class__
                        == transformers.GPTNeoXTokenizerFast
                        or blending_model_tokenizer.__class__
                        == transformers.GPT2TokenizerFast
                    )
                    and i == 0
                    and base_token.startswith(base_model_special_token)
                    and not blending_token.startswith(base_model_special_token)
                ):
                    blending_token = (
                        base_model_special_token + blending_token
                    )  # special case for mpt
                if vocab_align_type == "hard":
                    if (
                        base_token == blending_token
                    ):  # find the aligned mapping, use the corresponding logits
                        # the logits and indices at this step
                        for blending_logit, blending_index in zip(
                            blending_model_per_step_logits[j],
                            blending_model_per_step_indices[j],
                        ):
                            # the token corresponds to the logit and indices
                            blending_t = blending_model_tokenizer.convert_ids_to_tokens(
                                [blending_index]
                            )[0].replace(
                                blending_model_special_token, base_model_special_token
                            )
                            if blending_t in base_model_vocab:
                                aligned_index = base_model_vocab[
                                    blending_t
                                ]  # the index of the token in base model vocab
                                if (
                                    aligned_index
                                    not in aligned_blending_model_per_step_index
                                ):
                                    aligned_blending_model_per_step_index.append(
                                        aligned_index
                                    )
                                    aligned_blending_model_per_step_logit.append(
                                        blending_logit
                                    )
                    else:  # find error aligned mapping, use the one-hot logits
                        aligned_blending_model_per_step_index.append(
                            base_model_vocab[base_token]
                        )
                        aligned_blending_model_per_step_logit.append(1.0)
                elif vocab_align_type == "soft":
                    if (base_token == blending_token) or (
                        blending_token in blending_to_base_mapping
                        and base_token == blending_to_base_mapping[blending_token]
                    ):  # find the aligned mapping, use the corresponding logits
                        # the logits and indices at this step
                        for blending_logit, blending_index in zip(
                            blending_model_per_step_logits[j],
                            blending_model_per_step_indices[j],
                        ):  
                            # the token corresponds to the logit and indices
                            blending_t = blending_model_tokenizer.convert_ids_to_tokens(
                                [blending_index]
                            )[0].replace(
                                blending_model_special_token, base_model_special_token
                            )
                            blending_t = blending_to_base_mapping[blending_t]
                            if blending_t in base_model_vocab:
                                aligned_index = base_model_vocab[
                                    blending_t
                                ]  # the index of the token in base model vocab
                                if (
                                    aligned_index
                                    not in aligned_blending_model_per_step_index
                                ):
                                    aligned_blending_model_per_step_index.append(
                                        aligned_index
                                    )
                                    aligned_blending_model_per_step_logit.append(
                                        blending_logit
                                    )
                            else:
                                logging.warning(
                                    f"blending_t: {blending_t} not in base_model_vocab!"
                                )
                    else:  # find error aligned mapping, use the one-hot logits
                        aligned_blending_model_per_step_index.append(
                            base_model_vocab[base_token]
                        )
                        aligned_blending_model_per_step_logit.append(1.0)
                else:
                    logging.warning(
                        f"The vocab_align_type: '{vocab_align_type}' is not support!"
                    )
                    raise NotImplementedError
            else:  # one base token map to multiple blending token, in this case only fit base token. use the one-hot logits
                base_token = base_model_tokens[i]
                aligned_blending_model_per_step_index.append(base_model_vocab[base_token])
                aligned_blending_model_per_step_logit.append(1.0)
            aligned_blending_model_per_step_indices.append(
                aligned_blending_model_per_step_index
            )
            aligned_blending_model_per_step_logits.append(
                aligned_blending_model_per_step_logit
            )
        return (
            aligned_blending_model_per_step_logits,
            aligned_blending_model_per_step_indices,
        )
    
    def dtw(self, series_1, series_2, norm_func=np.linalg.norm):
        """Use dynamic time wrapping to align to tokenizers, modified from:
        https://github.com/talcs/simpledtw/blob/master/simpledtw.py"""
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
        i = matrix.shape[0] - 1
        j = matrix.shape[1] - 1
        matches = []
        mappings_series_1 = [list() for v in range(matrix.shape[0])]
        mappings_series_2 = [list() for v in range(matrix.shape[1])]
        while i > 0 or j > 0:
            matches.append((i, j))
            mappings_series_1[i].append(j)
            mappings_series_2[j].append(i)
            option_diag = matrix[i - 1, j - 1] if i > 0 and j > 0 else np.inf
            option_up = matrix[i - 1, j] if i > 0 else np.inf
            option_left = matrix[i, j - 1] if j > 0 else np.inf
            move = np.argmin([option_diag, option_up, option_left])
            if move == 0:
                i -= 1
                j -= 1
            elif move == 1:
                i -= 1
            else:
                j -= 1
        matches.append((0, 0))
        mappings_series_1[0].append(0)
        mappings_series_2[0].append(0)
        matches.reverse()
        for mp in mappings_series_1:
            mp.reverse()
        for mp in mappings_series_2:
            mp.reverse()

        return matches, matrix[-1, -1], mappings_series_1, mappings_series_2, matrix
    


    
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



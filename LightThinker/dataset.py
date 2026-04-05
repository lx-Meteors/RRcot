
import torch
from typing import List, Dict, Tuple, Union, Optional
from tqdm import tqdm
from copy import deepcopy
import time
import math
import random

from config import Config
from tokenizer import Tokenizer
from LightThinker.utils import create_attention_mask
from LightThinker.utils import visualize_labels, visualize_attention_mask
from LightThinker.utils import _print, read_jsonl, IGNORE_LABEL_ID, padding_item
from LightThinker.utils import create_attention_for_aug_data, create_attention_for_recover_data, create_attention_for_aug_data_apa_mtp

class MyDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file_path:str,
        config:Config,
        tokenizer: Tokenizer,
        padding_config: Dict,
        train_on_input:bool,
        change_rope: bool=False,
        output_compress_instruction:str=None,
        use_EPL=False,
    ):
        self.check_consistency = False
        self.train_on_input:bool = train_on_input
        self.change_rope = change_rope
        self.meta_data:List[Dict] = read_jsonl(
            file_path=file_path, progress=True
        )
        self.config:Config = config
        self.tokenizer:Tokenizer = tokenizer
        self.padding_config:Dict = padding_config
        
        # upper bound
        self.normal_data:List[Dict] = list()
        self.aug_data:List[Dict] = list()
        self.recover_data:List[Dict] = list()
        self.recover_prompt_data:List[Dict] = list()    
        self.aug_data_wo_prompt_comp:List[Dict] = list()    
        self.aug_data_wo_prompt_comp_apa_mtp:List[Dict] = list()

        self.output_compress_instruction:str = output_compress_instruction if output_compress_instruction != None else ""

        self.use_EPL = use_EPL
        self.output_step_sampling_cfg:Dict = deepcopy(self.config.output_step_sampling_cfg)
        
        # 预处理
        self.init()
        self.init_for_aug_data_wo_pc()

    def _estimate_step_importance(
        self,
        thought:str,
        running_freq:Dict[int, int],
    ) -> float:
        input_ids = self.tokenizer.tokenizer(
            thought,
            return_tensors=None,
            add_special_tokens=False,
        )['input_ids']
        if len(input_ids) == 0:
            return 0.0

        novelty_sum = 0.0
        for token_id in input_ids:
            novelty_sum += 1.0 / math.sqrt(running_freq.get(token_id, 0) + 1.0)

        uniq_ratio = len(set(input_ids)) / float(len(input_ids))
        length_bonus = math.log1p(len(input_ids))
        score = (novelty_sum / float(len(input_ids))) + 0.3 * uniq_ratio + 0.05 * length_bonus

        for token_id in input_ids:
            running_freq[token_id] = running_freq.get(token_id, 0) + 1
        return score

    def _tokenize_thought_steps(self, thoughts_list:List[str]) -> List[List[int]]:
        tokenized_steps:List[List[int]] = []
        for thought in thoughts_list:
            tokenized_steps.append(
                self.tokenizer.tokenizer(
                    thought,
                    return_tensors=None,
                    add_special_tokens=False,
                )['input_ids']
            )
        return tokenized_steps

    def _plan_step_memory_groups(self, thoughts_list:List[str]) -> Tuple[set, set, set]:
        total_steps = len(thoughts_list)
        if total_steps == 0:
            return set(), set(), set()

        cfg = self.output_step_sampling_cfg or {}
        if not cfg.get("enable", False):
            return set(range(total_steps)), set(), set()

        tokenized_steps = self._tokenize_thought_steps(thoughts_list)
        step_token_lens = [len(x) for x in tokenized_steps]

        # ===== recent token window =====
        recent_keep_tokens = int(cfg.get("recent_keep_tokens", -1))
        if recent_keep_tokens < 0:
            recent_keep = max(0, int(cfg.get("recent_keep", 1)))
            start_idx = max(0, total_steps - recent_keep)
            recent_keep_tokens = sum(step_token_lens[start_idx:])
        recent_keep_tokens = max(0, recent_keep_tokens)

        recent_steps:set = set()
        covered_recent_tokens = 0
        if recent_keep_tokens > 0:
            for step_idx in range(total_steps - 1, -1, -1):
                if step_token_lens[step_idx] == 0:
                    continue
                recent_steps.add(step_idx)
                covered_recent_tokens += step_token_lens[step_idx]
                if covered_recent_tokens >= recent_keep_tokens:
                    break

        # ===== training window =====
        training_window_size = int(cfg.get("training_window_size", -1))
        if training_window_size < 0:
            training_window_size = int(getattr(self.config, "structured_memory_cfg", {}).get("max_kv_budget", -1))
        strict_window = bool(cfg.get("strict_window", True))

        wait_budget_tokens = int(cfg.get("wait_budget_tokens", -1))
        if wait_budget_tokens < 0:
            wait_budget_tokens = int(getattr(self.config, "structured_memory_cfg", {}).get("wait_budget_tokens", 256))
        wait_budget_tokens = max(1, wait_budget_tokens)

        memory_budget_tokens = int(cfg.get("memory_budget_tokens", -1))
        if memory_budget_tokens < 0:
            memory_budget_tokens = int(getattr(self.config, "structured_memory_cfg", {}).get("memory_budget_tokens", 256))
        memory_budget_tokens = max(0, memory_budget_tokens)

        if training_window_size > 0:
            # 若 recent 已超过窗口，优先保留更近的 recent steps
            while len(recent_steps) > 0 and sum(step_token_lens[idx] for idx in recent_steps) > training_window_size:
                oldest_recent = min(recent_steps)
                recent_steps.remove(oldest_recent)

        history_steps = [idx for idx in range(total_steps) if idx not in recent_steps]

        # ===== global anchors (token-level sampling, step-level projection) =====
        global_anchor_mode = str(cfg.get("global_anchor_mode", "random")).lower()
        global_anchor_tokens = int(cfg.get("global_anchor_tokens", -1))
        if global_anchor_tokens < 0:
            global_anchor_keep = max(0, int(cfg.get("global_anchor_keep", 0)))
            valid_lens = [x for x in step_token_lens if x > 0]
            avg_len = (sum(valid_lens) // len(valid_lens)) if len(valid_lens) > 0 else 1
            global_anchor_tokens = global_anchor_keep * max(1, avg_len)
        global_anchor_tokens = max(0, global_anchor_tokens)

        anchor_step_weight:Dict[int, float] = dict()
        if global_anchor_tokens > 0 and len(history_steps) > 0:
            if global_anchor_mode == "heuristic":
                running_freq:Dict[int, int] = dict()
                for step_idx in history_steps:
                    anchor_step_weight[step_idx] = self._estimate_step_importance(thoughts_list[step_idx], running_freq)
            else:
                expanded_token_to_step:List[int] = []
                for step_idx in history_steps:
                    expanded_token_to_step.extend([step_idx] * step_token_lens[step_idx])
                if len(expanded_token_to_step) > 0:
                    n_sample = min(global_anchor_tokens, len(expanded_token_to_step))
                    sampled_positions = random.sample(range(len(expanded_token_to_step)), n_sample)
                    for pos in sampled_positions:
                        step_idx = expanded_token_to_step[pos]
                        anchor_step_weight[step_idx] = anchor_step_weight.get(step_idx, 0.0) + 1.0

        ranked_anchor_steps = sorted(
            anchor_step_weight.keys(),
            key=lambda idx: (
                anchor_step_weight[idx] / max(1, step_token_lens[idx]),
                anchor_step_weight[idx],
                -idx,
            ),
            reverse=True,
        )

        global_anchor_steps:set = set()
        if training_window_size > 0:
            recent_tokens = sum(step_token_lens[idx] for idx in recent_steps)
            remain_budget = max(0, training_window_size - recent_tokens)
            if strict_window:
                # 为 Memory 预留预算，避免训练窗口里 Recent+Global 把空间吃满
                remain_budget = max(0, remain_budget - memory_budget_tokens)
            remain_budget = min(remain_budget, global_anchor_tokens)
            for step_idx in ranked_anchor_steps:
                step_len = step_token_lens[step_idx]
                if step_len <= 0:
                    continue
                if step_len <= remain_budget:
                    global_anchor_steps.add(step_idx)
                    remain_budget -= step_len
        else:
            selected_tokens = 0
            for step_idx in ranked_anchor_steps:
                step_len = step_token_lens[step_idx]
                if step_len <= 0:
                    continue
                if selected_tokens + step_len > global_anchor_tokens:
                    continue
                global_anchor_steps.add(step_idx)
                selected_tokens += step_len

        if strict_window and training_window_size > 0:
            # 按 Wait budget 分批压缩：模拟 Recent 溢出 -> Wait 累计 -> 批压缩入 Memory
            selected_comp_steps:set = set()
            non_protected_steps = [
                idx for idx in range(total_steps)
                if idx not in recent_steps and idx not in global_anchor_steps
            ]

            wait_bucket:List[int] = []
            wait_tokens = 0
            memory_chunks:List[int] = []
            memory_tokens = 0

            for step_idx in non_protected_steps:
                step_len = step_token_lens[step_idx]
                if step_len <= 0:
                    selected_comp_steps.add(step_idx)
                    continue
                wait_bucket.append(step_idx)
                wait_tokens += step_len
                if wait_tokens >= wait_budget_tokens:
                    selected_comp_steps.update(wait_bucket)
                    comp_token_len = len(self.config.get_output_comp_token_id(cot_length=max(1, wait_tokens)))
                    memory_chunks.append(comp_token_len)
                    memory_tokens += comp_token_len
                    wait_bucket = []
                    wait_tokens = 0

                    # Memory FIFO 预算约束（与推理 memory 区上限一致）
                    if memory_budget_tokens > 0:
                        while memory_tokens > memory_budget_tokens and len(memory_chunks) > 0:
                            memory_tokens -= memory_chunks.pop(0)

            # 若尾部 Wait 未满但整体仍超训练窗口，强制触发一次尾部压缩
            if len(wait_bucket) > 0:
                recent_tokens = sum(step_token_lens[idx] for idx in recent_steps)
                global_tokens = sum(step_token_lens[idx] for idx in global_anchor_steps)
                tail_wait_tokens = sum(step_token_lens[idx] for idx in wait_bucket)
                estimated_total = recent_tokens + global_tokens + memory_tokens + tail_wait_tokens
                if estimated_total > training_window_size:
                    selected_comp_steps.update(wait_bucket)
                    tail_comp_len = len(self.config.get_output_comp_token_id(cot_length=max(1, tail_wait_tokens)))
                    memory_chunks.append(tail_comp_len)
                    memory_tokens += tail_comp_len
                    wait_bucket = []

                    if memory_budget_tokens > 0:
                        while memory_tokens > memory_budget_tokens and len(memory_chunks) > 0:
                            memory_tokens -= memory_chunks.pop(0)
        else:
            selected_comp_steps = self.tokenizer.sample_compressed_steps(
                total_steps=total_steps,
                step_sampling_cfg=cfg,
                excluded_steps=recent_steps | global_anchor_steps,
            )

        return selected_comp_steps, recent_steps, global_anchor_steps
    
    #token level，感觉没用到
    def insert_comp_for_output(self, output:str) -> Tuple[List[str], List[List[int]]]:
        assert self.config.output_comp_level == 'token'
        output_input_ids = self.tokenizer.tokenizer(
            output, return_tensors=None, add_special_tokens=False
        )['input_ids']
        input_ids_list:List[List[int]] = list()
        indicator_list:List[str] = list()
        step = self.config.output_comp_step

        comp_cnt = len(output_input_ids) // step 
        for i in range(comp_cnt):
            input_ids_list.append(
                output_input_ids[i*step: (i+1)*step]
            )
            indicator_list.append("abandoned")

            input_ids_list.append(
                self.output_compress_instruction + self.config.get_output_comp_token(return_list=False) + self.config.continue_token
            )
            indicator_list.append("compressed-output")
        if comp_cnt * step < len(output_input_ids):
            input_ids_list.append(
                output_input_ids[comp_cnt * step:len(output_input_ids)]
            )
            indicator_list.append("abandoned")
        # print(input_ids_list)
        # exit()
        return indicator_list, input_ids_list

    def insert_comp_for_prompt(
        self, 
        question:str, 
        question_list:List[str], 
        system_prompt:str, 
        system_prompt_list:List[str]
    ) -> Tuple[
        List[str],
        List[List[int]]
        # Union[NoneType, List[int]], 
        # List[int], 
        # Union[NoneType, List[int]]
    ]:
        prefix_input_ids = None
        middle_input_ids = None
        suffix_input_ids = None
        indicator_list:List[str] = list()
        input_ids_list:List[List[int]] = list()
        if self.config.prompt_save_template:
            prefix_input_ids = self.tokenizer.tokenizer(
                self.tokenizer.bos_token + self.config.template_cfg['prefix'],
                return_tensors=None,add_special_tokens=False
            )['input_ids']
            middle_input_ids = self.tokenizer.tokenizer(
                self.config.template_cfg['middle'],add_special_tokens=False
            )['input_ids']
            suffix_input_ids = self.tokenizer.tokenizer(
                self.config.template_cfg['suffix'],add_special_tokens=False
            )['input_ids']
        else:
            if not self.train_on_input:
                suffix_input_ids = [self.config.continue_token_id]
        if self.config.prompt_comp_level == 'token':
            step = self.config.prompt_comp_step
            if not self.config.prompt_save_template:
                prompt = self.config.template_cfg['prefix'] + \
                    system_prompt + self.config.template_cfg['middle'] + question + self.config.template_cfg['suffix']
                input_ids = [self.tokenizer.bos_token_id] + self.tokenizer.tokenizer(
                    prompt, return_tensors=None,add_special_tokens=False
                )['input_ids']
                for i in range(0, len(input_ids), step):
                    # ub = i+step
                    indicator_list.append("abandoned")
                    input_ids_list.append(input_ids[i:i+step])
                    indicator_list.append("compressed-prompt")
                    input_ids_list.append(self.config.get_prompt_comp_token_id())
                if suffix_input_ids != None:
                    indicator_list.append("save")
                    input_ids_list.append(suffix_input_ids)
            else:
                system_input_ids = self.tokenizer.tokenizer(system_prompt, return_tensors=None, add_special_tokens=False)['input_ids']
                prompt_input_ids = self.tokenizer.tokenizer(question, return_tensors=None, add_special_tokens=False)['input_ids']
                if prefix_input_ids != None:
                    indicator_list.append("save")
                    input_ids_list.append(prefix_input_ids)
                
                # ub = 0
                for i in range(0, len(system_input_ids), step):
                    # ub = i+step
                    indicator_list.append("abandoned")
                    input_ids_list.append(system_input_ids[i:i+step])
                    indicator_list.append("compressed-prompt")
                    input_ids_list.append(self.config.get_prompt_comp_token_id())

                if middle_input_ids != None:
                    indicator_list.append("save")
                    input_ids_list.append(middle_input_ids)
                # ub = 0
                for i in range(0, len(prompt_input_ids), step):
                    # ub = i+step
                    indicator_list.append("abandoned")
                    input_ids_list.append(prompt_input_ids[i:i+step])
                    indicator_list.append("compressed-prompt")
                    input_ids_list.append(self.config.get_prompt_comp_token_id())

                if suffix_input_ids != None:
                    indicator_list.append("save")
                    input_ids_list.append(suffix_input_ids)
        elif self.config.prompt_comp_level == 'sentence':
            #把[Prefix] [Sys Sent1] [Sys Sent2] [Middle] [Q Sent1] [Q Sent2] [Suffix]
            #压缩成[Prefix] [Comp Token] [Comp Token] [Middle] [Comp Token] [Comp Token] [Suffix]
            step = self.config.prompt_comp_step
            assert step == 1
            question_input_ids_list:List[List[int]] = list()
            system_prompt_input_ids_list:List[List[int]] = list()

            if not self.config.prompt_save_template:
                system_prompt_list[0] = self.config.template_cfg['prefix'] + system_prompt_list[0]
                system_prompt_list[-1] = system_prompt_list[-1] + self.config.template_cfg['middle']
                question_list[-1] = question_list[-1] + self.config.template_cfg['suffix']
                for q in question_list:
                    question_input_ids_list.append(
                        self.tokenizer.tokenizer(q, return_tensors=None, add_special_tokens=False)['input_ids']
                    )
                for s in system_prompt_list:
                    system_prompt_input_ids_list.append(
                        self.tokenizer.tokenizer(s, return_tensors=None, add_special_tokens=False)['input_ids']
                    )
                
                sentence_input_ids_list:List[List[int]] = system_prompt_input_ids_list + question_input_ids_list
                for i in range(0, len(sentence_input_ids_list)):
                    indicator_list.append("abandoned")
                    input_ids_list.append(sentence_input_ids_list[i])
                    indicator_list.append("compressed-prompt")
                    input_ids_list.append(self.config.get_prompt_comp_token_id())
            else:
                for q in question_list:
                    question_input_ids_list.append(
                        self.tokenizer.tokenizer(q, return_tensors=None, add_special_tokens=False)['input_ids']
                    )
                for s in system_prompt_list:
                    system_prompt_input_ids_list.append(
                        self.tokenizer.tokenizer(s, return_tensors=None, add_special_tokens=False)['input_ids']
                    )

                if prefix_input_ids != None:
                    indicator_list.append("save")
                    input_ids_list.append(prefix_input_ids)
                
                for i in range(0, len(system_prompt_input_ids_list)):
                    indicator_list.append("abandoned")
                    input_ids_list.append(system_prompt_input_ids_list[i])
                    indicator_list.append("compressed-prompt")
                    input_ids_list.append(self.config.get_prompt_comp_token_id())
                
                if middle_input_ids != None:
                    indicator_list.append("save")
                    input_ids_list.append(middle_input_ids)
                
                for i in range(0, len(question_input_ids_list)):
                    indicator_list.append("abandoned")
                    input_ids_list.append(question_input_ids_list[i])
                    indicator_list.append("compressed-prompt")
                    input_ids_list.append(self.config.get_prompt_comp_token_id())
                
                if suffix_input_ids != None:
                    indicator_list.append("save")
                    input_ids_list.append(suffix_input_ids)
        
        assert len(indicator_list) == len(input_ids_list)
        return indicator_list, input_ids_list

    def init(self):
        _print("initializing the dataset ...")
        self.config.convert2id(self.tokenizer)

        pbar = tqdm(total=len(self.meta_data))
        for item in self.meta_data:
            assert isinstance(item, dict)
            new_item = dict(
                meta_info=item,
                tokenized=dict()
            )

            question:str = item['question']
            system_prompt:str = item['system_prompt']
            system_prompt_list:List[str] = item['system_list']
            question_list:List[str] = item['question_list']
            thoughts_list:List[str] = item['thoughts_list']
            gt_output:str = item['gt_output']

            add_eos:bool = True
            if 'add_eos' in item:
                add_eos:bool = item['add_eos']
            # answer:str = f"\nTherefore, the answer is {item['answer']}"

            if len(thoughts_list) == 0 and len(question_list) == 0 and len(system_prompt_list) == 0:
                structured_input = [
                    self.tokenizer.bos_token + self.config.template_cfg['complete'].format(system=system_prompt, question=question),
                    gt_output + self.tokenizer.eos_token
                ]
            else:
                # structured_input is for normal data
                structured_input = [
                    self.tokenizer.bos_token + self.config.template_cfg['complete'].format(system=system_prompt, question=question),
                    gt_output + self.tokenizer.eos_token
                ]
                
            self.normal_data.append(
                dict(
                    tokenized=self.tokenizer.normal_data_tokenize(
                        structured_input=structured_input,
                        max_length=self.padding_config['max_length'],
                        train_on_input=self.train_on_input,
                        check_consistency=self.check_consistency,
                    ),
                    meta_info=item,
                )
            )
            #应该是在这里进行数据处理，并进行mask
            # 'save', 'abandoned', 'compressed-prompt', 'compressed-output'
            structured_input:List[List] = list()
            structured_input_indicator:List[List[str]] = list()
            mask_label_map:Dict = dict()
            
            mask_label_map[self.config.recover_token_id] = IGNORE_LABEL_ID
            mask_label_map[self.config.continue_token_id] = IGNORE_LABEL_ID
            for token_id in self.config.get_output_comp_token_id():
                mask_label_map[token_id] = IGNORE_LABEL_ID
            for token_id in self.config.get_prompt_comp_token_id():
                mask_label_map[token_id] = IGNORE_LABEL_ID

            # 1. insert prompt
            prompt_indicator_list, prompt_input_ids_list = self.insert_comp_for_prompt(
                question=question, question_list=question_list, 
                system_prompt=system_prompt, system_prompt_list=system_prompt_list
            )
            structured_input.append(prompt_input_ids_list)          # [n_turn, n_sent, n_token_per_sent]
            structured_input_indicator.append(prompt_indicator_list)    # [n_turn, n_sent]
            
            #AE任务的重建数据
            # 1.1 recover prompt part
            assert len(prompt_input_ids_list) == len(prompt_indicator_list)

            recover_prompt_input_ids = [self.config.recover_token_id]
            for input_ids, indicator in zip(prompt_input_ids_list, prompt_indicator_list):
                if indicator == 'abandoned':
                    recover_prompt_input_ids.extend(input_ids)
            recover_prompt_input_ids.append(self.tokenizer.eos_token_id)
            _, recover_prompt_item = self.tokenizer.aug_data_tokenize(
                structured_input=structured_input + [[recover_prompt_input_ids]],
                structured_input_indicator=structured_input_indicator + [["save"]],
                n_comp_for_prompt=self.config.prompt_comp_n_token,  
                n_continue_for_prompt=0,
                n_comp_for_output=self.config.output_comp_n_token,
                n_continue_for_output=1,
                mask_label_map=mask_label_map,
                max_length=self.padding_config['max_length'],
                train_on_input=self.train_on_input,
                check_consistency=self.check_consistency,
                recover_mode=False,
            )
            self.recover_prompt_data.append(
                dict(
                    meta_info=item,
                    tokenized=recover_prompt_item,
                )
            )
            

            # 2. output part
            if self.config.output_comp_level == 'token':
                output_indicator_list, output_content_list = self.insert_comp_for_output(
                    gt_output + self.tokenizer.eos_token if add_eos else gt_output
                )
                selected_comp_steps = set()
                recent_steps = set()
                global_anchor_steps = set()
            else:
                # sentence level
                output_indicator_list = list()
                output_content_list = list()
                selected_comp_steps, recent_steps, global_anchor_steps = self._plan_step_memory_groups(thoughts_list)
                for step_idx, thought in enumerate(thoughts_list):
                    if step_idx in selected_comp_steps:
                        output_indicator_list.append("abandoned")
                        output_content_list.append(thought + '\n' + self.config.split_token)
                        output_indicator_list.append("compressed-output")
                        output_content_list.append(self.output_compress_instruction + self.config.get_output_comp_token(return_list=False) + self.config.continue_token)
                    else:
                        output_indicator_list.append("save")
                        output_content_list.append(thought + '\n')
                if add_eos:
                    output_indicator_list.append("save")
                    output_content_list.append(self.tokenizer.eos_token)
            
            structured_input.append(output_content_list)
            structured_input_indicator.append(output_indicator_list)

            recover_item, aug_item = self.tokenizer.aug_data_tokenize(
                structured_input=structured_input,
                structured_input_indicator=structured_input_indicator,
                n_comp_for_prompt=self.config.prompt_comp_n_token,  
                n_continue_for_prompt=0,
                n_comp_for_output=self.config.output_comp_n_token,
                n_continue_for_output=1,
                mask_label_map=mask_label_map,
                max_length=self.padding_config['max_length'],
                train_on_input=self.train_on_input,
                check_consistency=self.check_consistency,
                recover_mode=True,
            )
            self.aug_data.append(
                dict(
                    meta_info=item,
                    tokenized={
                        **aug_item,
                        "selected_comp_steps": sorted(list(selected_comp_steps)) if self.config.output_comp_level == 'sentence' else [],
                        "recent_steps": sorted(list(recent_steps)) if self.config.output_comp_level == 'sentence' else [],
                        "global_anchor_steps": sorted(list(global_anchor_steps)) if self.config.output_comp_level == 'sentence' else [],
                        "total_thought_steps": len(thoughts_list),
                    },
                )
            )
            self.recover_data.append(
                recover_item
            )
            pbar.update(1)
    
    # 不压缩prompt
    def init_for_aug_data_wo_pc(self, system_compression: bool=False):
        pbar = tqdm(total=len(self.meta_data))
        for item in self.meta_data:
            assert isinstance(item, dict)
            new_item = dict(
                meta_info=item,
                tokenized=dict()
            )

            question:str = item['question']
            system_prompt:str = item['system_prompt']
            system_prompt_list:List[str] = item['system_list']
            question_list:List[str] = item['question_list']
            thoughts_list:List[str] = item['thoughts_list']
            gt_output:str = item['gt_output']
            add_eos:bool=True
            if 'add_eos' in item:
                add_eos:bool = item['add_eos']
            # we will not use the system_prompt_list and question_list

            structured_input:List[List] = list()
            structured_input_indicator:List[List[str]] = list()
            mask_label_map:Dict = dict()
            mask_label_map[self.config.recover_token_id] = IGNORE_LABEL_ID
            mask_label_map[self.config.continue_token_id] = IGNORE_LABEL_ID
            for token_id in self.config.get_output_comp_token_id():
                mask_label_map[token_id] = IGNORE_LABEL_ID
            for token_id in self.config.get_prompt_comp_token_id():
                mask_label_map[token_id] = IGNORE_LABEL_ID

            if system_compression == False:
                structured_input.append(
                    [
                        self.tokenizer.bos_token + \
                            self.config.template_cfg['complete'].format(
                                system=system_prompt, question=question
                            )
                    ]
                )
                structured_input_indicator.append(
                    ["save"]
                )
            else:
                assert False, "we don's support now"

            # 2.output
            output_comp_adaptive_num_token = list()
            if self.config.output_comp_level == 'token':
                output_indicator_list, output_content_list = self.insert_comp_for_output(
                    gt_output + self.tokenizer.eos_token if add_eos else gt_output
                )
                selected_comp_steps = set()
                recent_steps = set()
                global_anchor_steps = set()
            else:
                # sentence level
                output_indicator_list = list()
                output_content_list = list()
                selected_comp_steps, recent_steps, global_anchor_steps = self._plan_step_memory_groups(thoughts_list)
                for step_idx, thought in enumerate(thoughts_list):
                    if step_idx in selected_comp_steps:
                        output_indicator_list.append("abandoned")
                        output_content_list.append(thought + '\n' + self.config.split_token)
                        output_indicator_list.append("compressed-output")
                        if self.config.compression_ratio > 0: # compression_ratio大于0则自适应压缩，等于-1则固定token压缩个数
                            output_comp_tokens, num_comp_tokens = self.config.get_adaptive_output_comp_token(tokenizer=self.tokenizer, thought=thought)
                            output_content_list.append(self.output_compress_instruction + output_comp_tokens + self.config.continue_token)
                            output_comp_adaptive_num_token.append(num_comp_tokens)
                        else:
                            output_content_list.append(self.output_compress_instruction + self.config.get_output_comp_token(return_list=False) + self.config.continue_token)
                    else:
                        output_indicator_list.append("save")
                        output_content_list.append(thought + '\n')
                if add_eos:
                    output_indicator_list.append("save")
                    output_content_list.append(self.tokenizer.eos_token)
            
            structured_input.append(output_content_list)
            structured_input_indicator.append(output_indicator_list)

            recover_item, aug_item = self.tokenizer.aug_data_tokenize(
                structured_input=structured_input,
                structured_input_indicator=structured_input_indicator,
                n_comp_for_prompt=self.config.prompt_comp_n_token,  
                n_continue_for_prompt=0,
                n_comp_for_output=self.config.output_comp_n_token,
                n_continue_for_output=1,
                mask_label_map=mask_label_map,
                max_length=self.padding_config['max_length'],
                train_on_input=self.train_on_input,
                check_consistency=self.check_consistency,
                recover_mode=False,
                use_EPL=self.use_EPL,
                output_comp_adaptive_num_token=output_comp_adaptive_num_token
            )

            self.aug_data_wo_prompt_comp.append(
                dict(
                    meta_info=item,
                    tokenized={
                        **aug_item,
                        "selected_comp_steps": sorted(list(selected_comp_steps)) if self.config.output_comp_level == 'sentence' else [],
                        "recent_steps": sorted(list(recent_steps)) if self.config.output_comp_level == 'sentence' else [],
                        "global_anchor_steps": sorted(list(global_anchor_steps)) if self.config.output_comp_level == 'sentence' else [],
                        "total_thought_steps": len(thoughts_list),
                    }
                )
            )

            # 如果没有mtp配置就不处理mtp数据
            if self.config.mtp_cfg:
                recover_item, aug_item_register_mtp = self.tokenizer.aug_data_tokenize_apa_mtp(
                    structured_input=structured_input,
                    structured_input_indicator=structured_input_indicator,
                    n_comp_for_prompt=self.config.prompt_comp_n_token,  
                    n_continue_for_prompt=0,
                    n_comp_for_output=self.config.output_comp_n_token,
                    n_continue_for_output=1,
                    mask_label_map=mask_label_map,
                    max_length=self.padding_config['max_length'],
                    train_on_input=self.train_on_input,
                    check_consistency=self.check_consistency,
                    recover_mode=False,
                    use_EPL=self.use_EPL,
                    regitser_token=self.config.register_token,
                    output_comp_adaptive_num_token=output_comp_adaptive_num_token,
                    mtp_cfg=self.config.mtp_cfg
                )
            else:
                aug_item_register_mtp=None
            
            self.aug_data_wo_prompt_comp_apa_mtp.append(
                dict(
                    meta_info=item,
                    tokenized={
                        **aug_item_register_mtp,
                        "selected_comp_steps": sorted(list(selected_comp_steps)) if self.config.output_comp_level == 'sentence' else [],
                        "recent_steps": sorted(list(recent_steps)) if self.config.output_comp_level == 'sentence' else [],
                        "global_anchor_steps": sorted(list(global_anchor_steps)) if self.config.output_comp_level == 'sentence' else [],
                        "total_thought_steps": len(thoughts_list),
                    } if aug_item_register_mtp is not None else None
                )
            )


    def __getitem__(self, idx:int) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        return (
            self.normal_data[idx],
            self.aug_data[idx],
            self.recover_data[idx],
            self.recover_prompt_data[idx],
            self.aug_data_wo_prompt_comp[idx],
            self.aug_data_wo_prompt_comp_apa_mtp[idx]
        )
    
    def __len__(self) -> int:
        return len(self.normal_data)
    
class MyDataCollator:

    def __init__(
        self, 
        dataset:MyDataset,
        attention_config:Dict,
        exclude_continue:bool,
        sample_config:Dict,
    ):
        """
        attention_config:dict(
            diagonal=False,
            bi_directional=False,
            see_current=False,
            prefill_compress=True,  #
        )
        sample_config:dict(
            mode:str in [aug, normal]
            hybrid:bool=False,      
        )
        """
        self.dataset:MyDataset = dataset
        self.attention_config:Dict = attention_config
        self.exclude_continue:bool = exclude_continue
        self.sample_config = sample_config

    def _normal_mode(self, instances:List[Tuple]) -> Dict:
        final = dict(
            input_ids=list(),
            labels=list(),
        )
        for bsz_id, instance in enumerate(instances):
            normal_data, _, _, _, _, _ = instance
            new_item = padding_item(
                item=normal_data['tokenized'],
                padding_side=self.dataset.padding_config['padding_side'],
                label_padding_id=self.dataset.padding_config['label_padding_id'], 
                input_padding_id=self.dataset.padding_config['input_padding_id'], 
                max_length=self.dataset.padding_config['max_length'], 
                position_ids_padding_id=self.dataset.padding_config['position_ids_padding_id']
            )
            # print(new_item['input_ids'])
            final['input_ids'].append(new_item['input_ids'])
            final['labels'].append(new_item['labels'])
        
        # print(final['input_ids'])
        return dict(
            input_ids=torch.as_tensor(final['input_ids']),
            labels=torch.as_tensor(final['labels']),
        )
            
    def _aug_mode(self, instances:List[Tuple]) -> Dict:
        final = dict(
            input_ids=list(),
            labels=list(),
            attention_mask=list(),
            position_ids=list(),
            row_comp_index=list(),
            column_comp_index=list(),
        )

        # self.normal_data[idx],
        # self.aug_data[idx],
        # self.recover_data[idx]

        for bsz_id, instance in enumerate(instances):
            _, aug_data, _, _, _, _ = instance
            final['attention_mask'].append(
                create_attention_for_aug_data(
                    input_ids=aug_data['tokenized']['input_ids'],
                    locate_index_list=aug_data['tokenized']['locate_index'],
                    # [start, end, n_inst]
                    locate_indicator_list=aug_data['tokenized']['locate_indicator'],
                    bi_directional=self.attention_config['bi_directional'],
                    see_current=self.attention_config['see_current'],
                    diagonal=self.attention_config['diagonal'],
                    exclude_continue=self.exclude_continue,
                    max_length=self.dataset.padding_config['max_length'],
                    prefill_compress=self.attention_config['prefill_compress'],
                )
            )
            new_item = padding_item(
                item=aug_data['tokenized'],
                padding_side=self.dataset.padding_config['padding_side'],
                label_padding_id=self.dataset.padding_config['label_padding_id'], 
                input_padding_id=self.dataset.padding_config['input_padding_id'], 
                max_length=self.dataset.padding_config['max_length'], 
                position_ids_padding_id=self.dataset.padding_config['position_ids_padding_id']
            )
            final['input_ids'].append(new_item['input_ids'])
            final['labels'].append(new_item['labels'])
            final['position_ids'].append(new_item['position_ids'])
            # row and column
            for item in aug_data['tokenized']['locate_index']:
                start, end, l_inst, n_comp, n_continue = item
                # 直接添加，避免重复累积
                final['column_comp_index'].extend(
                    [end+l_inst+i for i in range(n_comp)]
                )
                final['row_comp_index'].extend([bsz_id] * n_comp)
        
        return dict(
            input_ids=torch.as_tensor(
                final['input_ids']
            ),
            labels=torch.as_tensor(
                final['labels']
            ),
            attention_mask=create_attention_mask(
                final['attention_mask'],
                dtype=torch.bfloat16
            ),
            position_ids=torch.as_tensor(
                final['position_ids']
            ),
            row_comp_index=torch.as_tensor(
                final['row_comp_index']
            ),
            column_comp_index=torch.as_tensor(
                final['column_comp_index']
            ),
        )
            
    def _recover_mode(self, instances:List[Tuple]) -> Dict:
        
        final = dict(
            input_ids=list(),
            labels=list(),
            attention_mask=list(),
            position_ids=list(),
            row_comp_index=list(),
            column_comp_index=list(),
        )

        for bsz_id, instance in enumerate(instances):
            _, aug_data, recover_data, _, _, _ = instance
            if len(recover_data) == 0:
                final['attention_mask'].append(
                    create_attention_for_aug_data(
                        input_ids=aug_data['tokenized']['input_ids'],
                        locate_index_list=aug_data['tokenized']['locate_index'],
                        # [start, end, n_inst]
                        locate_indicator_list=aug_data['tokenized']['locate_indicator'],
                        bi_directional=self.attention_config['bi_directional'],
                        see_current=self.attention_config['see_current'],
                        diagonal=self.attention_config['diagonal'],
                        exclude_continue=self.exclude_continue,
                        max_length=self.dataset.padding_config['max_length'],
                        prefill_compress=self.attention_config['prefill_compress'],
                    )
                )
                new_item = padding_item(
                    item=aug_data['tokenized'],
                    padding_side=self.dataset.padding_config['padding_side'],
                    label_padding_id=self.dataset.padding_config['label_padding_id'], 
                    input_padding_id=self.dataset.padding_config['input_padding_id'], 
                    max_length=self.dataset.padding_config['max_length'], 
                    position_ids_padding_id=self.dataset.padding_config['position_ids_padding_id']
                )
                final['input_ids'].append(new_item['input_ids'])
                final['labels'].append(new_item['labels'])
                final['position_ids'].append(new_item['position_ids'])

            else:
                new_input_ids = [self.dataset.config.recover_token_id]
                new_labels = [IGNORE_LABEL_ID]
                corresp_attention = list()
                for item in recover_data:
                    new_input_ids.extend(item['input_ids'])
                    new_labels.extend(item['labels'])
                    corresp_attention.append(item['corresp_attention'])
                    assert item['indicator'] == 'compressed-output'
                new_input_ids.append(self.dataset.tokenizer.eos_token_id)
                new_labels.append(self.dataset.tokenizer.eos_token_id)
                
                # print(aug_data['tokenized']['locate_index'])
                # print(aug_data['tokenized']['locate_indicator'])
                # exit()
                attention_mask, pos_offset = create_attention_for_recover_data(
                    input_ids=aug_data['tokenized']['input_ids'],
                    locate_index_list=aug_data['tokenized']['locate_index'],
                    # [start, end, n_inst]
                    locate_indicator_list=aug_data['tokenized']['locate_indicator'],
                    bi_directional=self.attention_config['bi_directional'],
                    see_current=self.attention_config['see_current'],
                    diagonal=self.attention_config['diagonal'],
                    exclude_continue=self.exclude_continue,
                    max_length=self.dataset.padding_config['max_length'],
                    added_input_ids=new_input_ids,
                    added_labels=new_labels,
                    added_corresp_attention=corresp_attention,
                    return_offset=True,
                    prefill_compress=self.attention_config['prefill_compress'],
                )
                final['attention_mask'].append(attention_mask)
                new_position_ids = [i+pos_offset for i in range(len(new_input_ids))]
                assert len(aug_data['tokenized']['input_ids'] + new_input_ids) == len(aug_data['tokenized']['position_ids'] + new_position_ids)
                
                new_item = padding_item(
                    item=dict(
                        input_ids=aug_data['tokenized']['input_ids'] + new_input_ids,
                        labels=aug_data['tokenized']['labels'] + new_labels,
                        position_ids=aug_data['tokenized']['position_ids'] + new_position_ids,
                    ),
                    padding_side=self.dataset.padding_config['padding_side'],
                    label_padding_id=self.dataset.padding_config['label_padding_id'], 
                    input_padding_id=self.dataset.padding_config['input_padding_id'], 
                    max_length=self.dataset.padding_config['max_length'], 
                    position_ids_padding_id=self.dataset.padding_config['position_ids_padding_id']
                )

                
                final['input_ids'].append(new_item['input_ids'])
                final['labels'].append(new_item['labels'])
                final['position_ids'].append(new_item['position_ids'])

            for item in aug_data['tokenized']['locate_index']:
                start, end, l_inst, n_comp, n_continue = item
                # 直接添加，避免重复累积
                final['column_comp_index'].extend(
                    [end+l_inst+i for i in range(n_comp)]
                )
                final['row_comp_index'].extend([bsz_id] * n_comp)

        return dict(
            input_ids=torch.as_tensor(
                final['input_ids']
            ),
            labels=torch.as_tensor(
                final['labels']
            ),
            attention_mask=create_attention_mask(
                final['attention_mask'],
                dtype=torch.bfloat16
            ),
            position_ids=torch.as_tensor(
                final['position_ids']
            ),
            row_comp_index=torch.as_tensor(
                final['row_comp_index']
            ),
            column_comp_index=torch.as_tensor(
                final['column_comp_index']
            ),
        )

    def _prompt_recover_mode(self, instances:List[Tuple]) -> Dict:
        # right padding
        final = dict(
            input_ids=list(),
            labels=list(),
            attention_mask=list(),
            position_ids=list(),
            row_comp_index=list(),
            column_comp_index=list(),
        )

        # self.normal_data[idx],
        # self.aug_data[idx],
        # self.recover_data[idx]
        # self.recover_prompt_data[idx]

        for bsz_id, instance in enumerate(instances):
            _, _, _, recover_prompt_data, _, _ = instance
            final['attention_mask'].append(
                create_attention_for_aug_data(
                    input_ids=recover_prompt_data['tokenized']['input_ids'],
                    locate_index_list=recover_prompt_data['tokenized']['locate_index'],
                    # [start, end, n_inst]
                    locate_indicator_list=recover_prompt_data['tokenized']['locate_indicator'],
                    bi_directional=self.attention_config['bi_directional'],
                    see_current=self.attention_config['see_current'],
                    diagonal=self.attention_config['diagonal'],
                    exclude_continue=self.exclude_continue,
                    max_length=self.dataset.padding_config['max_length'],
                    prefill_compress=self.attention_config['prefill_compress'],
                )
            )
            new_item = padding_item(
                item=recover_prompt_data['tokenized'],
                padding_side=self.dataset.padding_config['padding_side'],
                label_padding_id=self.dataset.padding_config['label_padding_id'], 
                input_padding_id=self.dataset.padding_config['input_padding_id'], 
                max_length=self.dataset.padding_config['max_length'], 
                position_ids_padding_id=self.dataset.padding_config['position_ids_padding_id']
            )
            final['input_ids'].append(new_item['input_ids'])
            final['labels'].append(new_item['labels'])
            final['position_ids'].append(new_item['position_ids'])
            # row and column
            for item in recover_prompt_data['tokenized']['locate_index']:
                start, end, l_inst, n_comp, n_continue = item
                # 直接添加，避免重复累积
                final['column_comp_index'].extend(
                    [end+l_inst+i for i in range(n_comp)]
                )
                final['row_comp_index'].extend([bsz_id] * n_comp)
        
        return dict(
            input_ids=torch.as_tensor(
                final['input_ids']
            ),
            labels=torch.as_tensor(
                final['labels']
            ),
            attention_mask=create_attention_mask(
                final['attention_mask'],
                dtype=torch.bfloat16
            ),
            position_ids=torch.as_tensor(
                final['position_ids']
            ),
            row_comp_index=torch.as_tensor(
                final['row_comp_index']
            ),
            column_comp_index=torch.as_tensor(
                final['column_comp_index']
            ),
        )
    
    
    def _aug_mode_wo_pc(self, instances:List[Tuple]) -> Dict:
        final = dict(
            input_ids=list(),
            labels=list(),
            attention_mask=list(),
            position_ids=list(),
            system_prompt_length=list(),
            row_comp_index=list(),
            column_comp_index=list(),
        )
        for bsz_id, instance in enumerate(instances):
            _, _, _, _, aug_data, _ = instance
            final['attention_mask'].append(
                create_attention_for_aug_data(
                    input_ids=aug_data['tokenized']['input_ids'],
                    locate_index_list=aug_data['tokenized']['locate_index'],
                    # [start, end, n_inst]
                    locate_indicator_list=aug_data['tokenized']['locate_indicator'],
                    bi_directional=self.attention_config['bi_directional'],
                    see_current=self.attention_config['see_current'],
                    diagonal=self.attention_config['diagonal'],
                    exclude_continue=self.exclude_continue,
                    max_length=self.dataset.padding_config['max_length'],
                    prefill_compress=False,
                )
            )
            new_item = padding_item(
                item=aug_data['tokenized'],
                padding_side=self.dataset.padding_config['padding_side'],
                label_padding_id=self.dataset.padding_config['label_padding_id'], 
                input_padding_id=self.dataset.padding_config['input_padding_id'], 
                max_length=self.dataset.padding_config['max_length'], 
                position_ids_padding_id=self.dataset.padding_config['position_ids_padding_id']
            )
            final['input_ids'].append(new_item['input_ids'])
            final['labels'].append(new_item['labels'])
            final['position_ids'].append(new_item['position_ids'])
            final['system_prompt_length'].append(new_item['system_prompt_length'])

            for item in aug_data['tokenized']['locate_index']:
                start, end, l_inst, n_comp, n_continue = item
                # 直接添加，避免重复累积
                final['column_comp_index'].extend(
                    [end+l_inst+i for i in range(n_comp)]
                )
                final['row_comp_index'].extend([bsz_id] * n_comp)


        return dict(
            input_ids=torch.as_tensor(
                final['input_ids']
            ),
            labels=torch.as_tensor(
                final['labels']
            ),
            attention_mask=create_attention_mask(
                final['attention_mask'],
                dtype=torch.bfloat16
            ),
            position_ids=torch.as_tensor(
                final['position_ids']
            ),
            system_prompt_length=torch.as_tensor(
                final['system_prompt_length']
            ),
            row_comp_index=torch.as_tensor(
                final['row_comp_index']
            ),
            column_comp_index=torch.as_tensor(
                final['column_comp_index']
            ),
        )

    def _aug_mode_wo_pc_apa_mtp(self, instances:List[Tuple]) -> Dict:
        final = dict(
            input_ids=list(),
            labels=list(),
            attention_mask=list(),
            position_ids=list(),
            system_prompt_length=list(),
            register_token_index=list(),
        )
        # 动态按当前 batch 的真实物理长度对齐（register 插入后长度会变化）
        batch_max_length = max(len(instance[5]['tokenized']['input_ids']) for instance in instances)

        for bsz_id, instance in enumerate(instances):
            _, _, _, _, _, aug_data = instance

            attention_mask, register_token_index = create_attention_for_aug_data_apa_mtp(
                    input_ids=aug_data['tokenized']['input_ids'],
                    locate_index_list=aug_data['tokenized']['locate_index'],
                    locate_indicator_list=aug_data['tokenized']['locate_indicator'],
                    exclude_continue=self.exclude_continue,
                    max_length=batch_max_length,
                    prefill_compress=False
                )
            final['attention_mask'].append(attention_mask)
            final['register_token_index'].append(register_token_index)
            new_item = padding_item(
                item=aug_data['tokenized'],
                padding_side=self.dataset.padding_config['padding_side'],
                label_padding_id=self.dataset.padding_config['label_padding_id'], 
                input_padding_id=self.dataset.padding_config['input_padding_id'], 
                max_length=batch_max_length, 
                position_ids_padding_id=self.dataset.padding_config['position_ids_padding_id']
            )
            final['input_ids'].append(new_item['input_ids'])
            final['labels'].append(new_item['labels'])
            final['position_ids'].append(new_item['position_ids'])
            final['system_prompt_length'].append(new_item['system_prompt_length'])


        return dict(
            input_ids=torch.as_tensor(
                final['input_ids']
            ),
            labels=torch.as_tensor(
                final['labels']
            ),
            attention_mask=create_attention_mask(
                final['attention_mask'],
                dtype=torch.bfloat16
            ),
            position_ids=torch.as_tensor(
                final['position_ids']
            ),
            system_prompt_length=torch.as_tensor(
                final['system_prompt_length']
            ),
            register_token_index=torch.as_tensor(
                final['register_token_index']
            ),
        )

    def __call__(self, instances:List[Tuple]) -> Dict:
        # 'aug', 'normal', 'recover'
        if self.sample_config['mode'] == 'aug':
            data = self._aug_mode(instances)
        elif self.sample_config['mode'] == 'recover':
            data = self._recover_mode(instances)
        elif self.sample_config['mode'] == 'normal':
            return self._normal_mode(instances)
        elif self.sample_config['mode'] == 'aug-wo-pc':
            return self._aug_mode_wo_pc(instances)
        elif self.sample_config['mode'] == 'aug-wo-pc-apa-mtp':
            return self._aug_mode_wo_pc_apa_mtp(instances)
        else:
            assert False
        if self.sample_config['hybrid'] == True:
            prompt_data = self._prompt_recover_mode(instances)
            return self._merge_dict(data, prompt_data)
        else:
            return data
    
    def _merge_dict(self, dict1, dict2) -> Dict:

        return dict(
            input_ids = torch.cat(
                (dict1['input_ids'], dict2['input_ids']), 
                dim=0
            ),
            labels=torch.cat(
                (dict1['labels'], dict2['labels']),
                dim=0
            ),
            attention_mask=torch.cat(
                (dict1['attention_mask'], dict2['attention_mask']),
                dim=0,
            ),
            position_ids=torch.cat(
                (dict1['position_ids'], dict2['position_ids']),
                dim=0,
            ),
            row_comp_index=torch.cat(
                (dict1['row_comp_index'], dict2['row_comp_index'] + len(dict1['input_ids'])),
                dim=0,
            ),
            column_comp_index=torch.cat(
                (dict1['column_comp_index'], dict2['column_comp_index']),
                dim=0,
            )
        )

        
if __name__ == '__main__':

    config_path = "/mnt/lxy/RRcot/configs/LightThinker/qwen/apa_mtp.json"
    tokenizer_path = "/mnt/zhaorunsong/models/Qwen2.5-0.5B-Instruct"
    dataset_path = "/mnt/lxy/RRcot/data/train/train_debug.jsonl"

    
    # bos_token="<|begin_of_text|>"
    # eos_token="<|eot_id|>"
    bos_token="<|im_start|>"
    eos_token="<|im_end|>"

    train_on_input = False
    exclude_continue = False

    config:Config = Config.from_file(config_path=config_path)
    # special_token_list:List[str] = config.special_token_name_list
    special_token_list:List[str] = list()

    tokenizer:Tokenizer = Tokenizer(
        tokenizer_path=tokenizer_path,
        bos_token=bos_token,
        eos_token=eos_token,
        special_token_list=None,
        add_prefix_space=False,
        change_rope=False,        
    )
    for token in config.special_token_name_list:
        if tokenizer.convert_tokens_to_ids(token) == None:
            special_token_list.append(token)
    if len(special_token_list) > 0:
        tokenizer.add_special_token(special_token_list)

    padding_config = dict(
        padding_side='right',
        label_padding_id=-100,
        input_padding_id=tokenizer.eos_token_id,
        max_length=1024,
        position_ids_padding_id=0,
    )
    attention_config = dict(
        diagonal=False,         
        bi_directional=False,
        see_current=True,
        prefill_compress=False,
    )
    sample_config = dict(
        mode=['aug', 'normal', 'recover', 'aug-wo-pc', 'aug-wo-pc-apa-mtp'][-1],
        hybrid=False
    )

    dataset = MyDataset(
        file_path=dataset_path,
        config=config,
        tokenizer=tokenizer,
        padding_config=padding_config,
        train_on_input=train_on_input,
        change_rope=False,
        output_compress_instruction="",
        use_EPL=True
    )

    data_collator = MyDataCollator(
        dataset=dataset,
        attention_config=attention_config,
        exclude_continue=exclude_continue,
        sample_config=sample_config,
    )

    batch = data_collator([dataset[0], dataset[1]])
    bsz_id = -1
    if 'attention_mask' in batch:
        visualize_attention_mask(
            attention_mask=batch['attention_mask'][bsz_id, 0].tolist(), 
            input_ids=batch['input_ids'][bsz_id].tolist(), 
            tokenizer=tokenizer,
            position_id=None,
            start_offset=0,
            end_offset=50,
        )
    else:
        print("no attention")

    print(
        visualize_labels(
            input_ids=batch['input_ids'][bsz_id].tolist(), 
            labels=batch['labels'][bsz_id].tolist(), 
            tokenizer=tokenizer,
            position_ids=batch['position_ids'][bsz_id].tolist() if 'position_ids' in batch else None
        )
    )


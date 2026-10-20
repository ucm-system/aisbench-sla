"""前缀缓存数据集生成。

移植并清理自参考工具 aisbench_auto_tools_prefix（MIT License, rayn-zzz）的
generate_dataset.py / data_picker.py，主要改动：

* 去掉 torch 依赖（原代码只用 torch.manual_seed 做种子，对 random 无影响）
* tokenizer 只加载一次（原实现每条样本加载一次，慢且重复）
* DataPicker 路径显式传入，不再依赖 cwd 下的 ./GSM8K.jsonl
* 生成失败直接抛异常，不再 exit(0)

数据集结构（与参考工具一致）：
    完整数据 = 公共前缀 + 3 个随机 token（--seed 控制） + 后缀
    前缀重复率 50% -> 每条请求前 50% 的 token 可以命中 KV cache
"""
from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- 选料
class DataPicker:
    """从参考语料里挑句子；prefix_flag=1 时不重复挑选并记录已用编号。"""

    def __init__(self, jsonl_file: str, record_file: str = "picked_ids.txt",
                 prefix_flag: int = 0) -> None:
        self.jsonl_file = jsonl_file
        self.record_file = record_file
        self.prefix_flag = prefix_flag
        self.total_lines = self._count_lines()
        if self.total_lines == 0:
            raise ValueError(f"参考语料为空：{jsonl_file}")
        self.picked_ids = self._load_picked_ids()

    def _count_lines(self) -> int:
        with open(self.jsonl_file, "r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)

    def _load_picked_ids(self) -> set:
        if os.path.exists(self.record_file):
            with open(self.record_file, "r", encoding="utf-8") as handle:
                return {int(line.strip()) for line in handle if line.strip()}
        return set()

    def _save_picked_ids(self) -> None:
        with open(self.record_file, "w", encoding="utf-8") as handle:
            for pid in sorted(self.picked_ids):
                handle.write(f"{pid}\n")

    def unpicked_ids(self) -> List[int]:
        return list(set(range(self.total_lines)) - self.picked_ids)

    def pick_one(self) -> Optional[str]:
        if self.prefix_flag == 1:
            available = self.unpicked_ids()
            if not available:
                return None
            selected = random.choice(available)
            self.picked_ids.add(selected)
            self._save_picked_ids()
        else:
            selected = random.randint(0, self.total_lines - 1)
        with open(self.jsonl_file, "r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index == selected:
                    return json.loads(line)["question"]
        return None


# --------------------------------------------------------------------- 工具
def load_tokenizer(tokenizer_path: str):
    """延迟导入 transformers —— 只有需要生成数据集时才要求该依赖。"""
    if not tokenizer_path:
        raise ValueError("生成数据集需要 tokenizer：请提供 --model-path")
    if not os.path.isdir(tokenizer_path):
        raise ValueError(f"tokenizer 路径不存在：{tokenizer_path}")
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - 环境相关
        raise RuntimeError(
            "未安装 transformers，无法生成数据集。"
            "请 pip install transformers，或直接提供现成数据集 --dataset <jsonl>"
        ) from exc
    logger.info("加载 tokenizer：%s", tokenizer_path)
    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def write_data(path: str, dataset: Sequence[str], num: Optional[int] = None) -> None:
    if num is not None:
        if len(dataset) < num:
            repeats = num // max(1, len(dataset))
            remainder = num % max(1, len(dataset))
            dataset = list(dataset) * repeats + list(dataset)[:remainder]
        else:
            dataset = list(dataset)[:num]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for item in dataset:
            handle.write(json.dumps({"question": item, "answer": "none"},
                                    ensure_ascii=False))
            handle.write("\n")


def _pad_or_truncate(tokenizer, tokens: List[int], target_len: int) -> List[int]:
    if len(tokens) >= target_len:
        return tokens[:target_len]
    if not tokens:
        return tokens
    repeat_times = (target_len + len(tokens) - 1) // len(tokens)
    return (tokens * repeat_times)[:target_len]


def create_dataset(tokenizer, picker: DataPicker, target_len: int,
                   number: int) -> Optional[List[str]]:
    """生成 number 条目标长度为 target_len 的文本（重复或截断参考语料）。"""
    output: List[str] = []
    attempts = 0
    max_attempts = max(100, number * 10)
    while len(output) < number and attempts < max_attempts:
        attempts += 1
        raw = picker.pick_one()
        if raw is None:
            break
        tokens = tokenizer.encode(raw, add_special_tokens=False)
        if not tokens:
            break
        tokens = _pad_or_truncate(tokenizer, tokens, target_len)
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        # 解码/再编码可能改变长度，修正一次
        if len(tokenizer.encode(text, add_special_tokens=False)) != target_len:
            tokens = _pad_or_truncate(
                tokenizer,
                tokenizer.encode(text, add_special_tokens=False),
                target_len)
            text = tokenizer.decode(tokens, skip_special_tokens=True)
        output.append(text)
    if len(output) < number:
        logger.error("生成数据集失败：可用语料不足（如反复失败，请清空 picked_ids.txt）")
        return None
    return output


def generate_unique_tokens(tokenizer, seed: int, n: int, number: int) -> List[str]:
    """生成 number 行、每行 n 个互不相同的随机 token 文本（seed 决定内容）。"""
    vocab_size = len(tokenizer)
    if n > vocab_size:
        raise ValueError(f"每行需要的 token 数 {n} 超过词表大小 {vocab_size}")
    lines: List[str] = []
    for line_index in range(number):
        line_seed = seed + line_index
        random.seed(line_seed)
        picked, seen = [], set()
        attempts, max_attempts = 0, n * 20
        while len(picked) < n and attempts < max_attempts:
            attempts += 1
            token_id = random.randint(0, vocab_size - 1)
            if token_id in seen:
                continue
            try:
                text = tokenizer.decode([token_id])
            except Exception:  # noqa: BLE001 - 个别 id 解码异常直接跳过
                continue
            picked.append(text)
            seen.add(token_id)
        if len(picked) < n:
            logger.warning("第 %d 行只生成了 %d/%d 个唯一 token", line_index + 1, len(picked), n)
        lines.append("".join(picked))
    return lines


def sample_target_length(rng: random.Random, fixed_length: int,
                         length_mean: Optional[int] = None,
                         length_std: Optional[float] = None,
                         length_min: Optional[int] = None,
                         length_max: Optional[int] = None) -> int:
    fixed_length = max(1, int(fixed_length))
    has_gauss = length_mean is not None and length_std is not None
    has_range = length_min is not None and length_max is not None
    low = 1 if length_min is None else max(1, int(length_min))
    high = None if length_max is None else max(1, int(length_max))
    if high is not None and low > high:
        low, high = high, low
    if has_gauss:
        mu = max(1, int(length_mean))
        sigma = max(0.0, float(length_std))
        value = mu if sigma == 0 else int(round(rng.gauss(mu, sigma)))
        if high is not None:
            value = min(value, high)
        return max(1, max(low, value))
    if has_range:
        return rng.randint(low, high)
    return fixed_length


def _length_tag(input_len: int, length_mean, length_std, length_min, length_max) -> str:
    if length_mean is not None and length_std is not None:
        tag = f"G{int(length_mean)}_{str(length_std).replace('.', 'd')}"
        if length_min is not None and length_max is not None:
            tag += f"_C{int(length_min)}_{int(length_max)}"
        return tag
    if length_min is not None and length_max is not None:
        return f"U{int(length_min)}_{int(length_max)}"
    return f"L{int(input_len)}"


def _truncate_or_pad_text(tokenizer, text: str, target_len: int) -> str:
    tokens = _pad_or_truncate(
        tokenizer, tokenizer.encode(text, add_special_tokens=False), target_len)
    return tokenizer.decode(tokens, skip_special_tokens=True)


# --------------------------------------------------------------------- 主流程
def create_multi_prefix_dataset(tokenizer_path: str, input_len: int, number: int,
                                save_path: str, prefix_flag: int, dp: int,
                                repeat_rate: float, seed: int, prefix_num: int,
                                length_mean: Optional[int] = None,
                                length_std: Optional[float] = None,
                                length_min: Optional[int] = None,
                                length_max: Optional[int] = None,
                                gsm8k_source: str = "",
                                ) -> Tuple[str, str]:
    """生成（前缀文件, 数据集文件）。prefix_flag=0 时不带前缀。"""
    os.makedirs(save_path, exist_ok=True)
    base_name = os.path.basename(os.path.normpath(tokenizer_path))
    tokenizer = load_tokenizer(tokenizer_path)
    source = gsm8k_source or default_gsm8k_source()
    picked_file = os.path.join(save_path, "picked_ids.txt")
    variable = ((length_mean is not None and length_std is not None)
                or (length_min is not None and length_max is not None))

    # ---------- 普通数据集 ----------
    if prefix_flag == 0:
        picker = DataPicker(source, picked_file, 0)
        if variable:
            rng = random.Random(seed)
            real_lens = [max(1, sample_target_length(rng, input_len, length_mean,
                                                     length_std, length_min, length_max))
                         for _ in range(number)]
            long_texts = create_dataset(tokenizer, picker, max(real_lens), number)
            if long_texts is None:
                raise RuntimeError("生成数据集失败：参考语料不足")
            dataset = [_truncate_or_pad_text(tokenizer, long_texts[i], real_lens[i])
                       for i in range(number)]
            tag = _length_tag(input_len, length_mean, length_std, length_min, length_max)
            dataset_path = os.path.join(
                save_path, f"GSM8K-{tag}-num{number}-{base_name}.jsonl")
            write_data(dataset_path, dataset, number)
            return "", dataset_path
        dataset = create_dataset(tokenizer, picker, input_len, number)
        if dataset is None:
            raise RuntimeError("生成数据集失败：参考语料不足")
        dataset_path = os.path.join(
            save_path, f"GSM8K-in{input_len}-num{number}-{base_name}.jsonl")
        write_data(dataset_path, dataset, number)
        return "", dataset_path

    # ---------- 前缀数据集 ----------
    picker = DataPicker(source, picked_file, 1)
    if variable:
        return _create_variable_prefix_dataset(
            tokenizer, picker, input_len, number, save_path, base_name, dp,
            repeat_rate, seed, prefix_num, length_mean, length_std,
            length_min, length_max)

    prefix_len = int(input_len * repeat_rate)
    prefix_pool = create_dataset(tokenizer, picker, prefix_len, prefix_num)
    if prefix_pool is None:
        raise RuntimeError("生成前缀失败：参考语料不足（可清空 picked_ids.txt 后重试）")
    prefix_dataset = [prefix_pool[i] for i in range(prefix_num) for _ in range(dp)]
    prefix_path = os.path.join(
        save_path, f"prefix-GSM8K-in{prefix_len}-num{dp * prefix_num}-{base_name}.jsonl")
    write_data(prefix_path, prefix_dataset, dp * prefix_num)

    if repeat_rate >= 1:
        dataset_path = os.path.join(
            save_path,
            f"GSM8K-in{prefix_len}-num{number}-{base_name}-repeatRate{repeat_rate}.jsonl")
        write_data(dataset_path, prefix_dataset, number)
        return prefix_path, dataset_path

    uniq_tokens = generate_unique_tokens(tokenizer, seed, 3, number)
    suffix_len = max(1, int(input_len - prefix_len - 3))
    suffix_pool = create_dataset(tokenizer, picker, suffix_len, number)
    if suffix_pool is None:
        raise RuntimeError("生成后缀失败：参考语料不足")
    dataset = [prefix_pool[i % prefix_num] + uniq_tokens[i] + suffix_pool[i]
               for i in range(number)]
    dataset_path = os.path.join(
        save_path,
        f"GSM8K-in{input_len}-num{number}-{base_name}-repeatRate{repeat_rate}.jsonl")
    write_data(dataset_path, dataset, number)
    return prefix_path, dataset_path


def _create_variable_prefix_dataset(tokenizer, picker: DataPicker, input_len: int,
                                    number: int, save_path: str, base_name: str,
                                    dp: int, repeat_rate: float, seed: int,
                                    prefix_num: int, length_mean, length_std,
                                    length_min, length_max) -> Tuple[str, str]:
    rng = random.Random(seed)
    real_lens = [max(1, sample_target_length(rng, input_len, length_mean, length_std,
                                             length_min, length_max))
                 for _ in range(number)]
    common_lens = [max(0, min(rl, int(round(rl * repeat_rate)))) for rl in real_lens]
    max_common = max(common_lens) if common_lens else 0

    if max_common > 0:
        prefix_pool = create_dataset(tokenizer, picker, max_common, prefix_num)
        if prefix_pool is None:
            raise RuntimeError("生成前缀失败：参考语料不足")
    else:
        prefix_pool = [""] * prefix_num

    prefix_dataset = [prefix_pool[i] for i in range(prefix_num) for _ in range(dp)]
    prefix_path = os.path.join(
        save_path, f"prefix-GSM8K-in{max_common}-num{dp * prefix_num}-{base_name}.jsonl")
    write_data(prefix_path, prefix_dataset, dp * prefix_num)

    if repeat_rate >= 1:
        dataset_path = os.path.join(
            save_path,
            f"GSM8K-in{max_common}-num{number}-{base_name}-repeatRate{repeat_rate}.jsonl")
        write_data(dataset_path, prefix_dataset, number)
        return prefix_path, dataset_path

    uniq_tokens = generate_unique_tokens(tokenizer, seed, 3, number)
    max_suffix = max(1, max(rl - cl - 3 for rl, cl in zip(real_lens, common_lens)))
    suffix_pool = create_dataset(tokenizer, picker, max_suffix, number)
    if suffix_pool is None:
        raise RuntimeError("生成后缀失败：参考语料不足")

    dataset = []
    for index in range(number):
        need = max(0, real_lens[index] - common_lens[index] - 3)
        prefix_text = prefix_pool[index % prefix_num]
        prefix_text = (_truncate_or_pad_text(tokenizer, prefix_text, common_lens[index])
                       if common_lens[index] > 0 and prefix_text else "")
        suffix_text = suffix_pool[index]
        suffix_text = (_truncate_or_pad_text(tokenizer, suffix_text, need)
                       if need > 0 and suffix_text else "")
        dataset.append(prefix_text + uniq_tokens[index] + suffix_text)

    tag = _length_tag(input_len, length_mean, length_std, length_min, length_max)
    dataset_path = os.path.join(
        save_path, f"GSM8K-{tag}-num{number}-{base_name}-repeatRate{repeat_rate}.jsonl")
    write_data(dataset_path, dataset, number)
    avg_hit = sum(c / r for c, r in zip(common_lens, real_lens)) / len(real_lens)
    logger.info("变长数据集：max_common_len=%d max_suffix_len=%d 平均命中比=%.2f%%",
                max_common, max_suffix, avg_hit * 100)
    return prefix_path, dataset_path


def default_gsm8k_source() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "GSM8K.jsonl")


def ensure_dataset(cfg) -> Tuple[str, str]:
    """返回（前缀文件, 数据集文件）；必要时生成，并按工作负载签名缓存复用。"""
    if cfg.dataset:
        if not os.path.isfile(cfg.dataset):
            raise ValueError(f"--dataset 指定的文件不存在：{cfg.dataset}")
        logger.info("使用指定数据集：%s", cfg.dataset)
        return "", cfg.dataset

    cache_file = os.path.join(cfg.dataset_path, ".dataset_cache.json")
    signature = cfg.workload_signature()
    cached = _read_cache(cache_file)
    if cached and cached.get("signature") == signature:
        prefix_path = cached.get("prefix_path", "")
        dataset_path = cached.get("dataset_path", "")
        if dataset_path and os.path.isfile(dataset_path):
            logger.info("复用已生成的数据集（签名一致）：%s", dataset_path)
            return prefix_path, dataset_path
        logger.warning("缓存记录的数据集不存在，重新生成：%s", dataset_path)

    prefix_path, dataset_path = create_multi_prefix_dataset(
        tokenizer_path=cfg.model_path,
        input_len=cfg.input_len,
        number=cfg.data_num,
        save_path=cfg.dataset_path,
        prefix_flag=1 if cfg.dataset_type == "prefix_cache" else 0,
        dp=cfg.dp,
        repeat_rate=cfg.repeat_rate,
        seed=cfg.seed,
        prefix_num=cfg.prefix_num,
        length_mean=cfg.length_mean,
        length_std=cfg.length_std,
        length_min=cfg.length_min,
        length_max=cfg.length_max,
        gsm8k_source=cfg.gsm8k_source,
    )
    _write_cache(cache_file, {"signature": signature, "prefix_path": prefix_path,
                              "dataset_path": dataset_path})
    logger.info("数据集就绪：%s（前缀文件：%s）", dataset_path, prefix_path or "无")
    return prefix_path, dataset_path


def ensure_seed_dataset(cfg, seed: int, save_path: str) -> Tuple[str, str]:
    """seed 专属数据集（每个探测点一份）：带签名缓存，重复运行可复用、不必重生成。"""
    cache_file = os.path.join(save_path, ".dataset_cache.json")
    signature = f"{cfg.workload_signature()}|seed={seed}"
    cached = _read_cache(cache_file)
    if (cached.get("signature") == signature and cached.get("dataset_path")
            and os.path.isfile(cached["dataset_path"])):
        logger.info("复用 seed=%s 的数据集：%s", seed, cached["dataset_path"])
        return cached.get("prefix_path", ""), cached["dataset_path"]

    prefix_path, dataset_path = create_multi_prefix_dataset(
        tokenizer_path=cfg.model_path,
        input_len=cfg.input_len,
        number=cfg.data_num,
        save_path=save_path,
        prefix_flag=1 if cfg.dataset_type == "prefix_cache" else 0,
        dp=cfg.dp,
        repeat_rate=cfg.repeat_rate,
        seed=seed,
        prefix_num=cfg.prefix_num,
        length_mean=cfg.length_mean,
        length_std=cfg.length_std,
        length_min=cfg.length_min,
        length_max=cfg.length_max,
        gsm8k_source=cfg.gsm8k_source,
    )
    _write_cache(cache_file, {"signature": signature, "prefix_path": prefix_path,
                              "dataset_path": dataset_path})
    return prefix_path, dataset_path


def _read_cache(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def _write_cache(path: str, payload: dict) -> None:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    except OSError as exc:  # 缓存写失败不影响主流程
        logger.warning("写数据集缓存失败（忽略）：%s", exc)

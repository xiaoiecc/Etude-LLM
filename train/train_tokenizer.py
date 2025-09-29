import os
import json
from glob import glob
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


class TokenizerConfig:
    # 你的源数据文件，现在可以包含各种结构的jsonl
    FILES_PATTERNS = [
        "training_data/pretrain/pretrain_hq.jsonl",
        "training_data/dpo/dpo.jsonl",
        # "training_data/other_format/*.jsonl" # 可以继续添加更多路径
    ]
    
    # 临时的文本缓存文件，用于训练
    CACHE_FILE = "training_data/tokenizer_cache.txt"
    
    VOCAB_SIZE = 16384

    SPECIAL_TOKENS = ["<|endoftext|>", "[UNK]", "<|im_end|>", "<|im_start|>"]
    
    # 最终分词器保存路径
    TOKENIZER_FILE = "weight/tokenizer.json" 

def extract_text_from_json(data):
    """
    一个递归生成器，用于从任何嵌套的JSON结构中提取所有字符串值。
    """
    if isinstance(data, str):
        # 如果当前数据就是字符串，直接返回它
        yield data
    elif isinstance(data, dict):
        # 如果是字典，遍历其所有的值
        for value in data.values():
            # 对每个值进行递归调用
            yield from extract_text_from_json(value)
    elif isinstance(data, list):
        # 如果是列表，遍历其所有元素
        for item in data:
            # 对每个元素进行递归调用
            yield from extract_text_from_json(item)

def universal_text_iterator(files_patterns: list):
    """
    一个更通用的生成器函数，逐行读取文件并解析jsonl，
    利用递归函数提取所有文本内容。
    """
    for pattern in files_patterns:
        files = glob(pattern)
        if not files:
            print(f"警告: 在 '{pattern}' 路径下未找到任何文件。")
            continue
        
        print(f"在 '{pattern}' 中找到 {len(files)} 个文件，开始提取文本...")
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    try:
                        # 解析JSON行
                        data = json.loads(line.strip())
                        # 使用递归函数从中提取所有文本
                        yield from extract_text_from_json(data)
                    except (json.JSONDecodeError, TypeError):
                        # 如果某行不是有效的JSON或出现其他错误，则跳过
                        continue

def prepare_cache_file(files_patterns: list, cache_file: str):
    """
    遍历所有jsonl文件，将提取的文本内容写入一个缓存txt文件，每段文本占一行。
    """
    print(f"开始创建用于训练的缓存文件: {cache_file}")
    
    cache_dir = os.path.dirname(cache_file)
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    with open(cache_file, 'w', encoding='utf-8') as f:
        # text_count = 0 # (可选) 如果想统计提取了多少段文本
        for text in universal_text_iterator(files_patterns):
            if text and text.strip():
                f.write(text.strip() + '\n')
                # text_count += 1
    
    # print(f"(可选) 共提取了 {text_count} 段文本。")
    print(f"缓存文件创建完成: {cache_file}")


def train_the_one_tokenizer():
    """
    主训练函数，逻辑保持不变。
    """
    cfg = TokenizerConfig()
    
    prepare_cache_file(cfg.FILES_PATTERNS, cfg.CACHE_FILE)

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(vocab_size=cfg.VOCAB_SIZE, special_tokens=cfg.SPECIAL_TOKENS)

    print("开始从缓存文件训练分词器 (内存优化模式)...")
    tokenizer.train([cfg.CACHE_FILE], trainer=trainer)
    print("训练结束")

    output_dir = os.path.dirname(cfg.TOKENIZER_FILE)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    tokenizer.save(cfg.TOKENIZER_FILE)
    print(f"分词器已经保存到: {cfg.TOKENIZER_FILE}")
    print(f"它的词表大小是: {tokenizer.get_vocab_size()}")

    print(f"正在删除缓存文件: {cfg.CACHE_FILE}")
    os.remove(cfg.CACHE_FILE)


if __name__ == "__main__":
    train_the_one_tokenizer()
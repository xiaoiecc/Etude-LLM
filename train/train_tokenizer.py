import os
import json
from glob import glob
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


class TokenizerConfig:
    FILES_PATTERNS = [
        "training_data/pretrain/pretrain_hq.jsonl",
        "training_data/dpo/dpo.jsonl",

    ]
    
    CACHE_FILE = "training_data/tokenizer_cache.txt"
    
    VOCAB_SIZE = 16384

    SPECIAL_TOKENS = ["<|endoftext|>", "[UNK]", "<|im_end|>", "<|im_start|>"]
    
    TOKENIZER_FILE = "weight/tokenizer.json" 

def extract_text_from_json(data):
    if isinstance(data, str):

        yield data
    elif isinstance(data, dict):

        for value in data.values():

            yield from extract_text_from_json(value)
    elif isinstance(data, list):

        for item in data:

            yield from extract_text_from_json(item)

def universal_text_iterator(files_patterns: list):
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

                        data = json.loads(line.strip())
                        yield from extract_text_from_json(data)
                    except (json.JSONDecodeError, TypeError):
                        continue

def prepare_cache_file(files_patterns: list, cache_file: str):
    print(f"创建缓存文件: {cache_file}")
    
    cache_dir = os.path.dirname(cache_file)
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    with open(cache_file, 'w', encoding='utf-8') as f:
        for text in universal_text_iterator(files_patterns):
            if text and text.strip():
                f.write(text.strip() + '\n')

    print(f"缓存文件创建完成: {cache_file}")


def train_the_one_tokenizer():
    cfg = TokenizerConfig()
    
    prepare_cache_file(cfg.FILES_PATTERNS, cfg.CACHE_FILE)

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(vocab_size=cfg.VOCAB_SIZE, special_tokens=cfg.SPECIAL_TOKENS)

    print("开始")
    tokenizer.train([cfg.CACHE_FILE], trainer=trainer)
    print("训练结束")

    output_dir = os.path.dirname(cfg.TOKENIZER_FILE)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    tokenizer.save(cfg.TOKENIZER_FILE)
    print(f"分词器已经保存到: {cfg.TOKENIZER_FILE}")
    print(f"词表大小: {tokenizer.get_vocab_size()}")
    os.remove(cfg.CACHE_FILE)


if __name__ == "__main__":
    train_the_one_tokenizer()
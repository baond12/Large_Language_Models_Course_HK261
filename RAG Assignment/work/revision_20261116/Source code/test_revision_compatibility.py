"""Offline CPU regression for the duplicate-revision error in Transformers 4.51.3.

Uses a tiny randomly initialized model, not Qwen or any assignment result.
"""
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, pipeline,
    __version__ as transformers_version,
)


def main():
    if transformers_version != '4.51.3':
        raise RuntimeError('This regression targets the observed Transformers 4.51.3 environment')
    torch.set_num_threads(1)
    torch.manual_seed(261)
    backend = Tokenizer(WordLevel({'[UNK]': 0, '[PAD]': 1, '[EOS]': 2, 'hello': 3, 'world': 4}, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', pad_token='[PAD]', eos_token='[EOS]')
    config = GPT2Config(vocab_size=5, n_positions=32, n_ctx=32, n_embd=8, n_layer=1, n_head=1, bos_token_id=2, eos_token_id=2, pad_token_id=1)
    model = GPT2LMHeadModel(config)
    caught = None
    try:
        pipeline('text-generation', model=model, tokenizer=tokenizer, device=-1, model_kwargs={'revision': 'fixed-test-revision'})
    except TypeError as exc:
        if "multiple values for keyword argument 'revision'" not in str(exc):
            raise
        caught = str(exc)
    if caught is None:
        raise AssertionError('Expected the observed duplicate-revision failure without an explicit framework')
    fixed = pipeline('text-generation', model=model, tokenizer=tokenizer, device=-1, model_kwargs={'revision': 'fixed-test-revision'}, framework='pt')
    result = fixed('hello', max_new_tokens=2, do_sample=False)
    assert isinstance(result[0]['generated_text'], str)
    report = {'scope': 'offline tiny-model CPU compatibility test, not assignment evaluation', 'transformers': transformers_version, 'torch': torch.__version__, 'failure_reproduced': caught, 'explicit_pt_pipeline_generates': True, 'qwen_on_P100_verified': False}
    out = Path(__file__).resolve().parent.parent / 'qa/revision_compatibility_validation.json'
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

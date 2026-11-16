"""Exercise the actual LangChain 0.3.1 method without downloading a model."""
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def main():
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.find_spec('langchain_huggingface')
    if spec is not None:
        assert importlib.metadata.version('langchain-huggingface') == '0.3.1'
        source_path = Path(spec.origin).parent / 'llms/huggingface_pipeline.py'
    else:
        source_path = root / 'qa/langchain_huggingface_0_3_1_pipeline_source.py'
        provenance = json.loads((root / 'qa/pipeline_source_provenance.json').read_text(encoding='utf-8'))
        assert hashlib.sha256(source_path.read_bytes()).hexdigest() == provenance['source_sha256']
    source = source_path.read_text(encoding='utf-8')
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'HuggingFacePipeline')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_generate')
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, method], type_ignores=[]))
    namespace = {'LLMResult': SimpleNamespace, 'Generation': SimpleNamespace}
    exec(compile(tree, str(source_path), 'exec'), namespace)
    generate = namespace['_generate']
    calls = []
    class Pipeline:
        task = 'text-generation'
        def __call__(self, prompts, **kwargs):
            calls.append(kwargs)
            return [{'generated_text': (prompt if kwargs.get('return_full_text', True) else '') + 'MODEL_COMPLETION'} for prompt in prompts]
    wrapper = SimpleNamespace(batch_size=4, pipeline=Pipeline())
    old = generate(wrapper, ['PROMPT'], max_new_tokens=256)
    assert calls[-1] == {}
    assert old.generations[0][0].text == 'PROMPTMODEL_COMPLETION'
    fixed = generate(wrapper, ['PROMPT'], pipeline_kwargs={'max_new_tokens': 256, 'return_full_text': False})
    assert calls[-1] == {'max_new_tokens': 256, 'return_full_text': False}
    assert fixed.generations[0][0].text == 'MODEL_COMPLETION'
    report = {'scope': 'actual LangChain 0.3.1 method with an observable pipeline, not a model-quality test', 'library_source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(), 'old_top_level_limit_not_forwarded_confirmed': True, 'fixed_256_limit_forwarded': True, 'fixed_completion_only_output': True, 'P100_Qwen_retry_verified': False}
    (root / 'qa/output_contract_validation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()

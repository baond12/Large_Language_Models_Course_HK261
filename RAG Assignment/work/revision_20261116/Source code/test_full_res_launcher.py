"""Check shell logging and error propagation with fake Python, never run a GPU job."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bash', default='bash')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory(prefix='launcher_', dir=root/'qa') as tmp:
        work = Path(tmp)/'work with spaces'
        code = work/'revision_20261116/Source code'
        code.mkdir(parents=True)
        script = code/'run_full_res.sh'
        shutil.copyfile(root/'Source code/run_full_res.sh',script)
        subprocess.run([args.bash,'-n',script.as_posix()],check=True,timeout=30)
        harness = '''python3() {
printf 'QA_FAKE_PYTHON_ARGS\\n'
printf '%s\\n' "$@"
printf 'QA_STDERR_CAPTURED\\n' >&2
return "${QA_FAKE_EXIT:-0}"
}
export -f python3
bash "$1" "$2"
'''
        for name,status in [('qa_success',0),('qa_failure',23)]:
            env = dict(os.environ,QA_FAKE_EXIT=str(status))
            result = subprocess.run([args.bash,'-c',harness,'qa',script.as_posix(),name],env=env,capture_output=True,text=True,timeout=30)
            assert result.returncode == status,(result.stdout,result.stderr)
            log = (work/f'artifacts/notebook_execution_20261116/{name}.console.log').read_text(encoding='utf-8')
            assert '--evaluation-set\nfull-res\n' in log
            assert '--limit' not in log
            assert 'QA_STDERR_CAPTURED' in log
            if status:
                assert 'completed successfully' not in result.stdout
    report = {'scope':'shell-launcher checks using fake Python in a temporary workspace','bash_syntax_valid':True,'spaces_in_paths_supported':True,'full_res_selected_without_limit':True,'stdout_and_stderr_logged':True,'failure_exit_code_preserved':True,'tmux_on_server_started':False,'GPU_job_started':False}
    (root/'qa/full_res_launcher_validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()

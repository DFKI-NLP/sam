import json
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime
from os.path import dirname, realpath, join, exists
from glob import glob

from allennlp import commands
from git import Repo, InvalidGitRepositoryError

IPFS_EXCEPTION = None
try:
    import ipfshttpclient
    ipfshttpclient.connect().close()
except Exception as e:
    IPFS_EXCEPTION = e
    pass


def get_run_states(path):
    if exists(path):
        with open(path) as f:
            lines = [l.strip()+'\n' for l in f.readlines() if l.strip() != '']
    else:
        lines = []
    return lines


def update_run_state(path, id_key, id_value, update_with):
    _run_states_lines = get_run_states(path=path)
    _run_states = OrderedDict()
    for l in lines:
        l_d = json.loads(l)
        _run_states[l_d[id_key]] = l_d
    _run_states[id_value].update(update_with)
    print(f'write {run_state_path} ...')
    with open(run_state_path, 'w') as f:
        f.writelines([json.dumps(l) + '\n' for l in _run_states.values()])


if __name__ == '__main__':
    # Setup logging
    #logging.basicConfig(
    #    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    #    datefmt="%m/%d/%Y %H:%M:%S",
    #    level=logging.DEBUG,
    #)

    dir_path = dirname(realpath(__file__))
    git_state_fn = 'runs_state.jsonl'
    run_state_path = join(dir_path, git_state_fn)
    time_start = datetime.now()
    KEY_TIME_START = 'time_stamp'
    run_state = {KEY_TIME_START: str(time_start), 'script': sys.argv[0], 'args': sys.argv[1:]}
    ipfs_add = []
    # add specified file(s) to ipfs (after command has finished)
    if '--ipfs-add' in sys.argv:
        if IPFS_EXCEPTION is not None:
            raise IPFS_EXCEPTION
        i = 1
        while i < len(sys.argv):
            if sys.argv[i] == '--ipfs-add':
                del sys.argv[i]
                fn_out_expanded = glob(sys.argv[i])
                ipfs_add.extend(fn_out_expanded)
                del sys.argv[i]
            else:
                i += 1
    # add the output depending on the command to ipfs
    if '--ipfs' in sys.argv:
        if IPFS_EXCEPTION is not None:
            raise IPFS_EXCEPTION
        # the format is: command -> ("argument to look for", "(sub-)path")
        CMD_ARGOUT_MAPPING = {
            # only add the model archive
            'train': ('-s', '/model.tar.gz'),
            'evaluate': ('--output-file', ''),
            'predagg': ('--output-file', '')
        }
        del sys.argv[sys.argv.index('--ipfs')]
        cmd = sys.argv[1]
        assert cmd in CMD_ARGOUT_MAPPING, f'unknown command: {cmd} (known: {CMD_ARGOUT_MAPPING.keys()})'
        arg_out, path = CMD_ARGOUT_MAPPING[cmd]
        assert arg_out in sys.argv, f'--ipfs requires an output, but {arg_out} is not set'
        idx_out = sys.argv.index(arg_out) + 1
        fn_out = sys.argv[idx_out] + path
        fn_out_expanded = glob(fn_out)
        ipfs_add.extend(fn_out_expanded)

    if len(ipfs_add) > 0:
        # dummy connect to check early if ipfs is available
        try:
            with ipfshttpclient.connect(addr='/dns/localhost/tcp/5001/http') as client:
                pass
        except ipfshttpclient.exceptions.ConnectionError as e:
            raise ConnectionError(f"could not connect to ipfs daemon (files to add: {ipfs_add})")
    try:
        repo = Repo(dir_path)
        sha = repo.head.object.hexsha
        run_state['git.hash'] = sha
        run_state['git.is_dirty'] = repo.is_dirty()
        run_state['git.branch'] = repo.active_branch.name
        print(f'INFO: git commit hash: {sha}')
        if repo.is_dirty():
            print(f'WARNING: git repo is DIRTY (the commit hash above does not completely track the current file state)')
        else:
            print(f'INFO: git repo is NOT dirty')
    except InvalidGitRepositoryError:
        print(f'WARNING: this is not a git repository, can not get commit hash: {dir_path}')

    try:
        reqs_bytes = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
        reqs = [l.strip() for l in reqs_bytes.decode('utf-8').split('\n') if l.strip() != '']
    except Exception as e:
        reqs = str(e)
    run_state['pip.freeze'] = reqs

    lines = get_run_states(run_state_path)
    lines.append(json.dumps(run_state) + '\n')
    print(f'write {run_state_path} ...')
    with open(run_state_path, 'w') as f:
        f.writelines(lines)

    try:
        commands.main()
    except Exception as e:
        update_run_state(path=run_state_path, id_key=KEY_TIME_START, id_value=str(time_start),
                         update_with={'time_delta': str(datetime.now() - time_start), 'exception': str(e)})
        raise e

    update_run_state(path=run_state_path, id_key=KEY_TIME_START, id_value=str(time_start),
                     update_with={'time_delta': str(datetime.now() - time_start)})
    if len(ipfs_add) > 0:
        cids = {}
        with ipfshttpclient.connect(addr='/dns/localhost/tcp/5001/http') as client:
            for fn_add in ipfs_add:
                print(f'add to ipfs: {fn_add}...')
                assert exists(fn_add), f'file to add to ipfs not found: {fn_add}'
                response = client.add(fn_add).as_json()
                cids[response['Hash']] = (fn_add, response["Size"])
                print(f'added: {response["Hash"]} -> {fn_add} (size: {response["Size"]})')
        update_run_state(path=run_state_path, id_key=KEY_TIME_START, id_value=str(time_start),
                         update_with={'cids': cids})

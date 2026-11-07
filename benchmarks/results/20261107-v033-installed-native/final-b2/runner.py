
import datetime, importlib.metadata as m, json, os, pathlib, subprocess, time, yaml
out = pathlib.Path('/evidence')
registry = yaml.safe_load(pathlib.Path('/work/.radiance/tests.yaml').read_text())
by_id = {t['id']:t for t in registry['tests']}
ids = ['r4d-attention-native','gdn-extreme-decay-native',
       'gdn-extreme-decay-h24-native','dflash-independent-noise-native',
       'nvfp4-native','v030-upstream-native','verify-head-native','lfm3-native']
if os.environ.get('RADIANCE_NATIVE_TEST_IDS'):
    ids=json.loads(os.environ['RADIANCE_NATIVE_TEST_IDS'])
assert all(by_id[i]['hardware']=='r9700' for i in ids)
# Release the HIP preflight process before launching isolated test processes.
subprocess.run(['python','-c','import pathlib,json\nout=pathlib.Path("/evidence")\nimport torch\nassert torch.cuda.is_available() and torch.cuda.device_count()==1\nprops=torch.cuda.get_device_properties(0)\nassert props.gcnArchName.startswith(\'gfx1201\'),props\n(torch.arange(16,device=\'cuda:0\')*2).cpu()\n(out/\'gpu-preflight.json\').write_text(json.dumps({\'count\':torch.cuda.device_count(),\n    \'name\':props.name,\'gcn_arch\':props.gcnArchName,\'total_memory\':props.total_memory,\n    \'hip\':torch.version.hip,\'render_node\':\'/dev/dri/renderD128\',\n    \'host_pci\':\'0000:03:00.0\'},indent=2)+\'\\n\')\n'],check=True)
components = ['torch','vllm','triton','amd-aiter','torchvision','transformers','xgrammar','flydsl','pytest']
before = {k:m.version(k) for k in components}
(out/'versions-before.json').write_text(json.dumps(before,indent=2)+'\n')
# pytest is supplied by the ROCm runtime requirements; test the unchanged image.
results=[]
for test_id in ids:
    test=by_id[test_id]
    env=os.environ.copy()
    env['DEVICE']='0'
    overrides={'DEVICE':'0'}
    if test_id=='nvfp4-native':
        overrides.update(RADIANCE_MXFP4='1',RADIANCE_MXFP4_W4A8='1')
    env.update(overrides)
    env['MODEL_PATH']='/models/lfm3' if test_id=='lfm3-image' else '/models/qwen'
    started=time.monotonic()
    with (out/f'{test_id}.log').open('w') as f:
        proc=subprocess.run(['bash','-lc','export PATH=/opt/vllm/bin:$PATH; '+test['command']],cwd='/work',env=env,stdout=f,stderr=subprocess.STDOUT)
    row={'id':test_id,'command':test['command'],'hardware':test['hardware'],'returncode':proc.returncode,'seconds':round(time.monotonic()-started,3),'environment_overrides':overrides}
    results.append(row)
    part=out/'results.tmp';part.write_text(json.dumps(results,indent=2)+'\n');part.replace(out/'results.json')
    print(json.dumps(row),flush=True)
    if proc.returncode and any(marker in (out/f'{test_id}.log').read_text()
        for marker in ('illegal memory access','hipErrorIllegalAddress',
                       'device-side assert','HSA_STATUS_ERROR')):
        print('Native device fault: remaining gates pending investigation',flush=True)
        break
raise SystemExit(any(r['returncode']!=0 for r in results))

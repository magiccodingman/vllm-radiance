
import datetime, importlib.metadata as m, json, os, pathlib, subprocess, time, yaml
out = pathlib.Path('/evidence')
registry = yaml.safe_load(pathlib.Path('/work/.radiance/tests.yaml').read_text())
by_id = {t['id']:t for t in registry['tests']}
ids = list(registry['profiles']['pr'])
ids += [t['id'] for t in registry['tests'] if t['hardware']=='candidate-image']
components = ['torch','vllm','triton','amd-aiter','torchvision','transformers','xgrammar','flydsl','pytest']
before = {k:m.version(k) for k in components}
(out/'versions-before.json').write_text(json.dumps(before,indent=2)+'\n')
# pytest is supplied by the ROCm runtime requirements; test the unchanged image.
results=[]
for test_id in ids:
    test=by_id[test_id]
    env=os.environ.copy()
    env['MODEL_PATH']='/models/lfm3' if test_id=='lfm3-image' else '/models/qwen'
    started=time.monotonic()
    with (out/f'{test_id}.log').open('w') as f:
        proc=subprocess.run(['bash','-lc','export PATH=/opt/vllm/bin:$PATH; '+test['command']],cwd='/work',env=env,stdout=f,stderr=subprocess.STDOUT)
    row={'id':test_id,'command':test['command'],'hardware':test['hardware'],'returncode':proc.returncode,'seconds':round(time.monotonic()-started,3)}
    results.append(row)
    part=out/'results.tmp';part.write_text(json.dumps(results,indent=2)+'\n');part.replace(out/'results.json')
    print(json.dumps(row),flush=True)
raise SystemExit(any(r['returncode']!=0 for r in results))

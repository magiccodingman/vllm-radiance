import datetime,hashlib,json,pathlib,subprocess,time,urllib.request
root=pathlib.Path(__file__).parent;out=root/'lfm3-official-conv-live-control-01';out.mkdir();m=json.loads((root/'lfm3-live-v2-01/manifest.json').read_text());cmd=m['command'];name='radiance-v033-lfm3-official-conv-control-01';cmd[cmd.index('--name')+1]=name
source=root/'lfm3-official-kernel-oracle-01/vllm/model_executor/layers/mamba/ops/causal_conv1d.py'
idx=cmd.index(m['image_id']);cmd[idx:idx]=['-v',str(source)+':/opt/vllm/lib/python3.12/site-packages/vllm/model_executor/layers/mamba/ops/causal_conv1d.py:ro']
m.update(command=cmd,purpose='negative control: official upstream convolution without Radiance FP32 products; all other installed runtime/checkpoint unchanged',override_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),started_at=datetime.datetime.now(datetime.timezone.utc).isoformat());(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
def post(prompt):
 body={'model':'radiance-v033-lfm3','prompt':prompt,'temperature':0,'seed':0,'max_tokens':16,'return_token_ids':True,'logprobs':5}
 req=urllib.request.Request('http://127.0.0.1:18034/v1/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=120) as r:return {'request':body,'response':json.load(r)}
try:
 m['container_id']=subprocess.check_output(cmd,text=True).strip();(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 for _ in range(240):
  try:
   with urllib.request.urlopen('http://127.0.0.1:18034/health',timeout=2) as r:assert r.status==200
   break
  except Exception:
   state=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]['State']
   if not state['Running']:raise RuntimeError('control container exited')
   time.sleep(2)
 else:raise TimeoutError('startup exceeded8min')
 prompt='The capital of France is';single=post(prompt);batch=post([prompt,prompt]);after=post(prompt)
 for label,value in [('single',single),('batch',batch),('after',after)]: (out/(label+'.json')).write_text(json.dumps(value,indent=2)+'\n')
 ids=single['response']['choices'][0]['token_ids'];result={'status':'CONTROL_COMPLETE','batch_equivalent':all(c['token_ids']==ids for c in batch['response']['choices']),'single_after_equal':after['response']['choices'][0]['token_ids']==ids};(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
finally:
 (out/'server.log').write_bytes(subprocess.check_output(['docker','logs',name],stderr=subprocess.STDOUT));subprocess.run(['docker','stop','--timeout','30',name],check=True)

import datetime,json,os,pathlib,subprocess,time,urllib.request
root=pathlib.Path(__file__).parent;repo=root.parent/'radiance-v033';out=root/'qwen-live-v2-02';results=[]
for _ in range(240):
 try:
  with urllib.request.urlopen('http://127.0.0.1:18033/health',timeout=2) as r:assert r.status==200
  break
 except Exception:
  state=json.loads(subprocess.check_output(['docker','inspect','radiance-v033-qwen-v2-02'],text=True))[0]['State']
  if not state['Running']:raise RuntimeError('server exited')
  time.sleep(2)
else:raise TimeoutError('server not healthy after8min')
(out/'startup-ready.log').write_bytes(subprocess.check_output(['docker','logs','radiance-v033-qwen-v2-02'],stderr=subprocess.STDOUT))
python=str(root.parent/'radiance-upgrade-env/bin/python')
commands=[('resident',[python,str(repo/'benchmarks/bin/check_resident_platform.py'),'--url','http://127.0.0.1:18033','--model','radiance-v033-qwen','--out',str(out/'resident'),'--vision']),('sampled-tool',['bash',str(repo/'benchmarks/bin/run_tool_schema_gate.sh')]),('open-object',[python,str(repo/'benchmarks/bin/run_open_object_tool_gate.py'),'--base-url','http://127.0.0.1:18033/v1','--model','radiance-v033-qwen','--runs','5','--capture-root',str(out/'open-object')])]
env={**os.environ,'BASE_URL':'http://127.0.0.1:18033/v1','MODEL_NAME':'radiance-v033-qwen','ATTEMPTS':'30','RUN_DIR':str(out/'sampled-tool')}
for name,command in commands:
 with (out/(name+'.log')).open('x') as log:p=subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
 results.append({'id':name,'command':command,'returncode':p.returncode,'finished_at':datetime.datetime.now(datetime.timezone.utc).isoformat()});(out/'live-results.json').write_text(json.dumps(results,indent=2)+'\n')
 print(name,p.returncode,flush=True)
 # Keep independent wire evidence even if a trajectory/schema gate fails.
 if json.loads(subprocess.check_output(['docker','inspect','radiance-v033-qwen-v2-02'],text=True))[0]['State']['Running'] is not True:break
raise SystemExit(any(row['returncode'] for row in results))

import datetime,json,pathlib,subprocess,time,urllib.request
root=pathlib.Path(__file__).parent;out=root/'qwen-dflash-tp1-01';name='radiance-v033-qwen-dflash-tp1-01';base='http://127.0.0.1:18036';rows=[]
def save(name,value):(out/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
def post(name,endpoint,body):
 req=urllib.request.Request(base+endpoint,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=180) as r:x=json.load(r)
 save(name,{'request':body,'response':x});return x
try:
 for _ in range(300):
  try:
   with urllib.request.urlopen(base+'/health',timeout=2) as r:assert r.status==200
   break
  except Exception:
   state=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]['State'];assert state['Running'],state;time.sleep(2)
 else:raise TimeoutError('DFlash startup exceeded10min')
 (out/'startup-ready.log').write_bytes(subprocess.check_output(['docker','logs',name],stderr=subprocess.STDOUT))
 x=post('owners','/collective_rpc',{'method':'platform_snapshot'})['results'][0]
 assert x['speculative_method']=='dflash' and x['num_speculative_steps']==7 and x['draft_model']['class']=='DFlash2Qwen3ForCausalLM'
 assert 'Fp8LinearMethod' in json.dumps(x['draft_owners'])
 rows.append({'gate':'actual-native-fp8-draft-owner','pass':True})
 common={'model':'radiance-v033-qwen','temperature':0,'seed':0,'max_tokens':32,'return_token_ids':True}
 body={**common,'prompt':'The capital of France is'};first=post('short','/v1/completions',body);ids=first['choices'][0]['token_ids'];assert ids and 'Paris'in first['choices'][0]['text']
 streambody={**body,'stream':True,'stream_options':{'include_usage':True}};req=urllib.request.Request(base+'/v1/completions',data=json.dumps(streambody).encode(),headers={'Content-Type':'application/json'});raw=[];events=[];streamids=[];done=False
 with urllib.request.urlopen(req,timeout=180) as r:
  for line in r:
   raw.append(line.decode())
   if not line.startswith(b'data: '):continue
   data=line[6:].strip()
   if data==b'[DONE]':done=True;break
   event=json.loads(data);events.append(event)
   for choice in event['choices']:streamids.extend(choice.get('token_ids')or[])
 (out/'short-stream.sse').write_text(''.join(raw));save('short-stream',{'request':streambody,'events':events,'done':done});assert done and streamids==ids
 rows.append({'gate':'short-sse-token-parity','pass':True})
 tools=[{'type':'function','function':{'name':'lookup','strict':True,'parameters':{'type':'object','properties':{'city':{'type':'string','enum':['Paris']}},'required':['city'],'additionalProperties':False}}}]
 tool=post('tool','/v1/chat/completions',{'model':common['model'],'temperature':0,'seed':0,'max_tokens':128,'chat_template_kwargs':{'enable_thinking':False},'messages':[{'role':'user','content':'Use lookup for Paris.'}],'tools':tools,'tool_choice':{'type':'function','function':{'name':'lookup'}}})
 call=tool['choices'][0]['message']['tool_calls'];assert len(call)==1 and call[0]['function']['name']=='lookup' and json.loads(call[0]['function']['arguments'])=={'city':'Paris'};rows.append({'gate':'named-tool','pass':True})
 with urllib.request.urlopen(base+'/metrics',timeout=10) as r:metrics=r.read().decode()
 (out/'metrics.prom').write_text(metrics)
 drafts=sum(float(line.rsplit(' ',1)[1]) for line in metrics.splitlines() if line.startswith('vllm:spec_decode_num_draft_tokens_total{'))
 accepted=sum(float(line.rsplit(' ',1)[1]) for line in metrics.splitlines() if line.startswith('vllm:spec_decode_num_accepted_tokens_total{'))
 assert drafts>0 and accepted>0,(drafts,accepted);rows.append({'gate':'actual-draft-and-acceptance','pass':True,'draft_tokens':drafts,'accepted_tokens':accepted})
 save('result',{'status':'PASS','gates':rows,'scope':'boundedTP1FP8DFlash7smoke only; no matched-performance, TP2 or strict speculative-versus-serial equivalence claim'});print('PASS',flush=True)
except BaseException as e:
 save('result',{'status':'FAIL','gates':rows,'error':str(e),'type':type(e).__name__});raise
finally:
 (out/'server-final.log').write_bytes(subprocess.check_output(['docker','logs',name],stderr=subprocess.STDOUT));subprocess.run(['docker','stop','--timeout','30',name],check=True)

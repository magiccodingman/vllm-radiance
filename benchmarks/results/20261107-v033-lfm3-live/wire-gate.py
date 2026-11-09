import datetime,json,pathlib,time,urllib.error,urllib.request
root=pathlib.Path(__file__).parent/'lfm3-live-v2-01';out=root/'wire-04';out.mkdir();url='http://127.0.0.1:18034';model='radiance-v033-lfm3';rows=[]
def save(name,value):
 (out/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
def post(name,endpoint,body):
 start=time.monotonic();request=urllib.request.Request(url+endpoint,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 try:
  with urllib.request.urlopen(request,timeout=120) as response:result=json.load(response)
 except urllib.error.HTTPError as exc:
  save(name,{'request':body,'status':exc.code,'body':exc.read().decode()});raise
 save(name,{'request':body,'response':result,'seconds':time.monotonic()-start});return result

def stream(name,endpoint,body):
 body={**body,'stream':True,'stream_options':{'include_usage':True}}
 request=urllib.request.Request(url+endpoint,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'});events=[];raw=[];done=False
 with urllib.request.urlopen(request,timeout=120) as response:
  for line in response:
   raw.append(line.decode())
   if not line.startswith(b'data: '):continue
   value=line[6:].strip()
   if value==b'[DONE]':done=True;break
   events.append(json.loads(value))
 (out/(name+'.sse')).write_text(''.join(raw));save(name,{'request':body,'events':events,'done':done});assert done and events
 return events

def passed(name,**details):
 rows.append({'gate':name,'pass':True,**details});save('progress',rows)

common={'model':model,'temperature':0,'seed':0,'max_tokens':16,'return_token_ids':True}
try:
 owners=post('owners','/collective_rpc',{'method':'platform_snapshot'});serialized=json.dumps(owners)
 assert 'Lfm3ForCausalLM' in serialized and 'CompressedTensorsLinearMethod' in serialized and 'vllm.v1.worker.gpu.model_runner' in serialized
 assert 'Selected TritonFp8BlockScaledMMKernel for CompressedTensorsW8A8Fp8' in (root/'startup-before-wire02.log').read_text()
 passed('native-owners')
 prompt='The capital of France is';body={**common,'prompt':prompt}
 first=post('short','/v1/completions',body);ids=first['choices'][0]['token_ids'];assert ids and first['usage']['completion_tokens']==len(ids) and 'Paris' in first['choices'][0]['text']
 events=stream('short-stream','/v1/completions',body);stream_ids=[token for event in events for choice in event['choices'] for token in (choice.get('token_ids') or [])];assert stream_ids==ids
 passed('short-sse-token-parity',tokens=len(ids))
 long_prompt=('A red apple sits beside a blue cup. '*70)+'\nThe capital of France is';tokens=post('long-tokenize','/tokenize',{'model':model,'prompt':long_prompt})['tokens'];assert len(tokens)>256
 long_body={**common,'prompt':long_prompt};long1=post('long','/v1/completions',long_body);long2=post('long-repeat','/v1/completions',long_body);assert long1['choices'][0]['token_ids']==long2['choices'][0]['token_ids'] and long1['usage']['prompt_tokens']>256
 passed('long-window-repeat',prompt_tokens=len(tokens))
 batch=post('batch','/v1/completions',{**common,'prompt':[prompt,prompt]});assert len(batch['choices'])==2
 batch_equal=all(choice['token_ids']==ids for choice in batch['choices'])
 rows.append({'gate':'batch-token-equivalence','pass':batch_equal});save('progress',rows)
 after=post('short-after-long','/v1/completions',body);assert after['choices'][0]['token_ids']==ids
 passed('state-isolation-after-batch')
 chat={'model':model,'temperature':0,'seed':0,'max_tokens':128,'chat_template_kwargs':{'enable_thinking':False}}
 schema={'type':'object','properties':{'answer':{'type':'object','properties':{'city':{'type':'string','enum':['Paris']}},'required':['city'],'additionalProperties':False}},'required':['answer'],'additionalProperties':False}
 structured=post('structured','/v1/chat/completions',{**chat,'messages':[{'role':'user','content':'Return Paris in answer.city as JSON.'}],'response_format':{'type':'json_schema','json_schema':{'name':'city','strict':True,'schema':schema}}});assert json.loads(structured['choices'][0]['message']['content'])=={'answer':{'city':'Paris'}}
 passed('nested-json-schema')
 tools=[{'type':'function','function':{'name':'add','description':'Add two integers. Synthetic test only.','strict':True,'parameters':{'type':'object','properties':{'a':{'type':'integer','enum':[2]},'b':{'type':'integer','enum':[3]}},'required':['a','b'],'additionalProperties':False}}}]
 messages=[{'role':'system','content':'Use add to calculate 2+3. After receiving its result, answer with the result.'},{'role':'user','content':'Calculate 2 + 3 using add.'}]
 tool_body={**chat,'messages':messages,'tools':tools,'tool_choice':{'type':'function','function':{'name':'add'}}}
 result=post('tool','/v1/chat/completions',tool_body);choice=result['choices'][0];calls=choice['message']['tool_calls'];assert choice['finish_reason']=='stop' and len(calls)==1
 assert calls[0]['function']['name']=='add' and json.loads(calls[0]['function']['arguments'])=={'a':2,'b':3}
 events=stream('tool-stream','/v1/chat/completions',tool_body);collected={};finishes=[]
 for event in events:
  for choice in event['choices']:
   if choice.get('finish_reason'):finishes.append(choice['finish_reason'])
   for call in choice.get('delta',{}).get('tool_calls') or []:
    entry=collected.setdefault(call['index'],{'name':'','arguments':''});function=call.get('function',{});entry['name']+=function.get('name') or '';entry['arguments']+=function.get('arguments') or ''
 assert len(collected)==1 and finishes==['stop']
 entry=next(iter(collected.values()));assert entry['name']=='add' and json.loads(entry['arguments'])=={'a':2,'b':3}
 passed('named-tool-and-sse')
 assistant={'role':'assistant','content':result['choices'][0]['message'].get('content'),'tool_calls':calls}
 final=post('tool-result','/v1/chat/completions',{**chat,'messages':messages+[assistant,{'role':'tool','tool_call_id':calls[0]['id'],'content':'5'}],'tools':tools,'tool_choice':'none'});assert '5' in (final['choices'][0]['message']['content'] or '')
 passed('tool-result-roundtrip')
 status='PASS' if all(row['pass'] for row in rows) else 'FAIL'
 save('result',{'status':status,'scope':'bounded TP1 native FP8 LFM3 serving, no TP2/capacity/performance claim','gates':rows,'finished_at':datetime.datetime.now(datetime.timezone.utc).isoformat()})
 print(json.dumps({'status':status,'gates':rows}),flush=True)
 if status!='PASS':raise SystemExit(1)
except Exception as exc:
 save('result',{'status':'FAIL','error':str(exc),'type':type(exc).__name__,'completed_gates':rows});raise

from pathlib import Path
exec(compile(Path('/helpers/run_raw_hip.py').read_text().split('count=I();',1)[0],'qualified-raw-hip-helpers','exec'))
import radiance_mxfp4_fp8 as extension
root=Path('/out');count=I();hip('hipGetDeviceCount',c.byref(count));assert count.value==1
hip('hipSetDevice',0);meta=json.loads((root/'fixtures.json').read_text());records=[]
for row in meta['cases']:
 for folded in (True,False):
  with np.load(root/row['file']) as f:data={k:np.ascontiguousarray(f[k]) for k in f.files}
  m,n,k=row['m'],row['n'],row['k'];guard=np.uint16(np.float32(37).view(np.uint32)>>16)
  data['slab']=np.full(m*n+512,guard,dtype=np.uint16)
  data['partial']=np.full(4*m*n+512,37.,dtype=np.float32)
  data['counter']=np.zeros((n+127)//128+8+16,dtype=np.int32)
  data['counter'][:8]=37;data['counter'][-8:]=37
  buffers={name:Buffer(data[name]) for name in ('a','w','ws','wref','ascale','slab','partial','counter')}
  extension.set_decode_scratch(buffers['partial'].ptr.value+1024,4*m*n*4,buffers['counter'].ptr.value+32)
  stream=P();hip('hipStreamCreate',c.byref(stream))
  def call():
   extension.launch(buffers['a'].ptr.value,buffers['w'].ptr.value,buffers['ws'].ptr.value,
        buffers['wref'].ptr.value if folded else 0,buffers['ascale'].ptr.value,
        buffers['slab'].ptr.value+512,m,n,k,stream.value)
  def check(sign):
   hip('hipDeviceSynchronize')
   for name in ('slab','partial','counter'):buffers[name].read(data[name])
   assert np.all(data['slab'][:256]==guard) and np.all(data['slab'][-256:]==guard)
   assert np.all(data['partial'][:256]==37.) and np.all(data['partial'][-256:]==37.)
   assert np.all(data['counter'][:8]==37) and np.all(data['counter'][-8:]==37)
   assert np.all(data['counter'][8:-8]==0),'decode counters were not reset'
   actual=(data['slab'][256:-256].astype(np.uint32)<<16).view(np.float32).astype(np.float64).reshape(m,n)
   error=actual-sign*data['expected'];relative=float(np.linalg.norm(error)/np.linalg.norm(data['expected']))
   assert np.isfinite(actual).all()
   assert relative<.01,(row,folded,relative)
   assert np.max(np.abs(error))<np.max(np.abs(data['expected']))*.02+1e-4
   return relative
  for _ in range(3):call()
  error=check(1)
  graph=P();execution=P();hip('hipStreamBeginCapture',stream,0);call();hip('hipStreamEndCapture',stream,c.byref(graph))
  hip('hipGraphInstantiate',c.byref(execution),graph,None,None,0)
  changed=np.ascontiguousarray(data['a']^np.uint8(0x80));buffers['a'].write(changed)
  for _ in range(3):hip('hipGraphLaunch',execution,stream)
  graph_error=check(-1)
  hip('hipGraphExecDestroy',execution);hip('hipGraphDestroy',graph);hip('hipStreamDestroy',stream)
  for buffer in buffers.values():buffer.close()
  records.append({**row,'folded':folded,'relative_error':error,'changed_input_graph_error':graph_error,
                  'guards_intact':True,'decode_counters_reset':True})
  (root/'progress.json').write_text(json.dumps({'complete':False,'cases':records},indent=2)+'\n')
  print(row['file'],'folded',folded,'PASS',error,'graph',graph_error,flush=True)
(root/'result.json').write_text(json.dumps({'status':'ROCM10_NATIVE_MXFP4_PASS_NOT_FRAMEWORK_QUALIFICATION',
 'complete':True,'visible_count':count.value,'scope':'Direct HIP on GPU0, exact FP8 activation/E2M1 weight/E8M0-scale operands versus independent FP64 dots. Folded and per-block kernels, decode/prefill/TN4, K64-only width, partial N, TP1-wide N34816, guards/counters/changed-input graphs. No PyTorch loaded in GPU process.',
 'extension_sha256':hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),'cases':records},indent=2)+'\n')
print('ROCm10 MXFP4/FP8: 16 FP64 oracle/guard/counter/changed-input graph cases PASS')

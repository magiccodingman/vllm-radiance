# Reuse the already-qualified raw HIP memory/graph bindings. This process loads
# NumPy and the compiled maintained extension, never PyTorch.
from pathlib import Path
helpers=Path('/helpers/run_raw_hip.py').read_text().split('count=I();',1)[0]
exec(compile(helpers,'qualified-raw-hip-helpers','exec'))

root=Path('/out')
count=I();hip('hipGetDeviceCount',c.byref(count));assert count.value==1,count.value
hip('hipSetDevice',0)
meta=json.loads((root/'fixtures.json').read_text());records=[]
for row in meta['cases']:
 with np.load(root/row['file']) as f:data={k:np.ascontiguousarray(f[k]) for k in f.files}
 tokens,heads,hg=row['tokens'],row['heads'],row['query_heads'];n=len(row['lengths'])
 guard=np.uint16(np.float32(37).view(np.uint32)>>16)
 data['matrix']=np.zeros((tokens,heads,64),dtype=np.uint16)
 data['slab']=np.full(tokens*heads*128+512,guard,dtype=np.uint16)
 data['final']=np.zeros(data['expected_state'].shape,dtype=np.float32)
 keys=('q','k','v','g','beta','h0','cu','matrix','slab','final')
 buffers={key:Buffer(data[key]) for key in keys}
 stream=P();hip('hipStreamCreate',c.byref(stream))
 r4d.gdn_kkt_solve_k128_c64_bf16(buffers['k'].ptr.value,buffers['beta'].ptr.value,
     buffers['g'].ptr.value,buffers['matrix'].ptr.value,buffers['cu'].ptr.value,
     n,tokens,heads,hg,128,64,stream.value)
 def call():
  r4d.gdn_chunk_scan_k128_v128_c64_bf16(
      buffers['q'].ptr.value,buffers['k'].ptr.value,buffers['v'].ptr.value,
      buffers['matrix'].ptr.value,buffers['g'].ptr.value,buffers['beta'].ptr.value,
      buffers['h0'].ptr.value,buffers['slab'].ptr.value+512,buffers['final'].ptr.value,
      buffers['cu'].ptr.value,n,heads,hg,128,128,64,128**-.5,stream.value)
 def check(sign):
  hip('hipDeviceSynchronize');buffers['slab'].read(data['slab']);buffers['final'].read(data['final'])
  assert np.all(data['slab'][:256]==guard) and np.all(data['slab'][-256:]==guard)
  actual=(data['slab'][256:-256].astype(np.uint32)<<16).view(np.float32).astype(np.float64).reshape(data['expected'].shape)
  out_error=float(np.linalg.norm(actual-sign*data['expected'])/np.linalg.norm(data['expected']))
  state_error=float(np.linalg.norm(data['final'].astype(np.float64)-data['expected_state'])/np.linalg.norm(data['expected_state']))
  assert np.isfinite(actual).all() and np.isfinite(data['final']).all()
  assert max(out_error,state_error)<.01,(row,out_error,state_error)
  return out_error,state_error
 call();out_error,state_error=check(1)
 graph=P();execution=P();hip('hipStreamBeginCapture',stream,0);call();hip('hipStreamEndCapture',stream,c.byref(graph))
 hip('hipGraphInstantiate',c.byref(execution),graph,None,None,0)
 changed=np.ascontiguousarray(data['q']^np.uint16(0x8000));buffers['q'].write(changed)
 for _ in range(3):hip('hipGraphLaunch',execution,stream)
 graph_out,graph_state=check(-1)
 hip('hipGraphExecDestroy',execution);hip('hipGraphDestroy',graph);hip('hipStreamDestroy',stream)
 for buffer in buffers.values():buffer.close()
 records.append({**row,'output_relative_error':out_error,'state_relative_error':state_error,
                 'changed_query_graph_output_error':graph_out,'changed_query_graph_state_error':graph_state,
                 'guards_intact':True,'finite':True})
 (root/'progress.json').write_text(json.dumps({'complete':False,'cases':records},indent=2)+'\n')
 print(row['file'],'PASS',out_error,state_error,'graph',graph_out,graph_state,flush=True)
result={'status':'ROCM10_NATIVE_GDN_PASS_NOT_FRAMEWORK_QUALIFICATION','complete':True,
 'scope':'Direct HIP/pybind calls on one visible authorized GPU, both 48/16 and 24/8 geometries. Exact original extreme-decay fixture cases and one-percent numerical gates; no PyTorch loaded in GPU process. Not distributed TP2 qualification.',
 'visible_count':count.value,'r4d_version':r4d.__version__,'maintained_commit':r4d.__maintained_commit__,
 'extension_sha256':hashlib.sha256(Path(r4d.__file__).read_bytes()).hexdigest(),'cases':records}
(root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print('ROCm10 maintained GDN: 24 FP64 extreme-decay/guard/changed-input graph cases PASS')

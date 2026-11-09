import ctypes as c
import hashlib,json
from pathlib import Path
import numpy as np
import r4d

root=Path('/out')
h=c.CDLL('/opt/rocm/lib/libamdhip64.so')
P=c.c_void_p;I=c.c_int;S=c.c_size_t
signatures={'hipGetDeviceCount':[c.POINTER(I)],'hipSetDevice':[I],
 'hipMalloc':[c.POINTER(P),S],'hipFree':[P],'hipMemcpy':[P,P,S,I],
 'hipDeviceSynchronize':[],'hipStreamCreate':[c.POINTER(P)],'hipStreamDestroy':[P],
 'hipStreamBeginCapture':[P,I],'hipStreamEndCapture':[P,c.POINTER(P)],
 'hipGraphInstantiate':[c.POINTER(P),P,P,P,S],'hipGraphLaunch':[P,P],
 'hipGraphExecDestroy':[P],'hipGraphDestroy':[P]}
for name,args in signatures.items():
 fn=getattr(h,name);fn.argtypes=args;fn.restype=I
h.hipGetErrorString.argtypes=[I];h.hipGetErrorString.restype=c.c_char_p

def hip(name,*args):
 status=getattr(h,name)(*args)
 if status:raise RuntimeError(f'{name}: {status} {h.hipGetErrorString(status).decode()}')

class Buffer:
 def __init__(self,array):
  self.ptr=P();self.bytes=array.nbytes
  hip('hipMalloc',c.byref(self.ptr),array.nbytes)
  self.write(array)
 def write(self,array):
  assert array.flags.c_contiguous and array.nbytes==self.bytes
  hip('hipMemcpy',self.ptr,P(array.ctypes.data),array.nbytes,1)
 def read(self,array):
  assert array.flags.c_contiguous and array.nbytes==self.bytes
  hip('hipMemcpy',P(array.ctypes.data),self.ptr,array.nbytes,2)
 def close(self):hip('hipFree',self.ptr)

count=I();hip('hipGetDeviceCount',c.byref(count));assert count.value==1,count.value
hip('hipSetDevice',0)
meta=json.loads((root/'fixtures.json').read_text())
records=[]
for row in meta['cases']:
 with np.load(root/row['file']) as f: data={k:np.ascontiguousarray(f[k]) for k in f.files}
 q=data['q'];n,qlen,qheads,width=q.shape;kvheads=4;maximum=max(row['contexts'])
 selected=dict(head_dim=256,gqa=6,block_size=16,causal=1,q_dtype='bf16',
               kv_dtype='bf16' if row['dtype']=='bf16' else 'fp8_e4m3')
 operation='attn_decode_paged' if qlen<=10 else 'attn_prefill_paged'
 if qlen<=10:selected['q_len']=qlen
 name=r4d.select(operation,**selected);assert name
 launch=getattr(r4d,name)
 size=int(r4d.attn_decode_h256_gqa6_scratch_bytes(n,min(qlen,10),qheads,kvheads,256,maximum,0))
 guard=np.uint16(np.float32(37).view(np.uint32)>>16)
 data['slab']=np.full(q.size+512,guard,dtype=np.uint16)
 data['scratch']=np.zeros(max(1,size),dtype=np.uint8)
 buffers={key:Buffer(data[key]) for key in ('q','kv','table','lengths','ks','vs','slab','scratch')}
 stream=P();hip('hipStreamCreate',c.byref(stream))
 def call():
  launch(buffers['q'].ptr.value,buffers['kv'].ptr.value,buffers['table'].ptr.value,
         buffers['lengths'].ptr.value,buffers['slab'].ptr.value+512,
         buffers['ks'].ptr.value,buffers['vs'].ptr.value,buffers['scratch'].ptr.value,
         n,qlen,qheads,kvheads,256,16,data['table'].shape[1],kvheads*16*512,16*512,
         256**-.5,0,maximum,stream.value)
 def check(expected):
  hip('hipDeviceSynchronize');buffers['slab'].read(data['slab'])
  assert np.all(data['slab'][:256]==guard) and np.all(data['slab'][-256:]==guard)
  out=(data['slab'][256:-256].astype(np.uint32)<<16).view(np.float32).astype(np.float64).reshape(q.shape)
  error=out-expected
  relative=float(np.linalg.norm(error)/np.linalg.norm(expected))
  maximum_error=float(np.max(np.abs(error)))
  assert np.isfinite(out).all()
  assert relative<.01,(row,relative)
  assert maximum_error<np.max(np.abs(expected))*.02+1e-4,(row,maximum_error)
  return out,relative,maximum_error
 for _ in range(3):call()
 out,relative,maximum_error=check(data['expected'])
 graph=P();execution=P()
 hip('hipStreamBeginCapture',stream,0);call();hip('hipStreamEndCapture',stream,c.byref(graph))
 hip('hipGraphInstantiate',c.byref(execution),graph,None,None,0)
 buffers['q'].write(data['changed']);hip('hipGraphLaunch',execution,stream)
 changed,graph_error,_=check(data['expected_changed'])
 assert not np.array_equal(out,changed)
 hip('hipGraphExecDestroy',execution);hip('hipGraphDestroy',graph);hip('hipStreamDestroy',stream)
 for b in buffers.values():b.close()
 records.append({**row,'kernel':name,'relative_error':relative,'maximum_error':maximum_error,
                 'changed_input_graph_error':graph_error})
 print(row['file'],'PASS',relative,'graph',graph_error,flush=True)
result={'status':'ROCM10_NATIVE_KERNEL_PASS_NOT_FRAMEWORK_QUALIFICATION',
 'scope':'Direct HIP/pybind calls on one visible authorized GPU. CPU fixtures use independent FP64 attention over exact stored cache values. No PyTorch loaded in the GPU process.',
 'visible_count':count.value,'r4d_version':r4d.__version__,'maintained_commit':r4d.__maintained_commit__,
 'extension_sha256':hashlib.sha256(Path(r4d.__file__).read_bytes()).hexdigest(),'cases':records}
(root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print('ROCm10 maintained paged attention: 12 oracle/guard/changed-input graph cases PASS')

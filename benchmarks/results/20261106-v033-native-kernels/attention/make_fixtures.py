import ast,json
from pathlib import Path
import numpy as np
import torch
root=Path('/out')
source=Path('/repo/benchmarks/bin/check_r4d_attention_native.py')
f=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='reference')
scope={'torch':torch};exec(compile(ast.Module(body=[f],type_ignores=[]),'independent-reference','exec'),scope)
reference=scope['reference']
torch.manual_seed(3306)
rows=[]
for dtype_name,dtype in (('bf16',torch.bfloat16),('fp8',torch.float8_e4m3fn)):
 for qlen,lengths in [(1,[17,33]),(7,[65,129]),(10,[1025,769]),(17,[129,193]),(65,[257,193]),(129,[513,385])]:
  n,heads,width=len(lengths),4,256;blocks=(max(lengths)+15)//16
  table=(torch.randperm(n*blocks)+1).int().view(n,blocks)
  kv=(torch.randn((1+n*blocks,heads,16,2*width))*.5).to(dtype)
  q=torch.randn((n,qlen,heads*6,width),dtype=torch.bfloat16)
  ks=torch.full((n,heads),.75);vs=torch.full((n,heads),1.25)
  expected=reference(q,kv,table,lengths,ks,vs).numpy()
  changed=(q*1.5).contiguous();expected_changed=reference(changed,kv,table,lengths,ks,vs).numpy()
  file=f'{dtype_name}-q{qlen}.npz'
  np.savez(root/file,q=q.view(torch.uint16).numpy(),changed=changed.view(torch.uint16).numpy(),
    kv=kv.view(torch.uint8 if dtype_name=='fp8' else torch.uint16).numpy(),table=table.numpy(),
    lengths=np.array(lengths,dtype=np.int32),ks=ks.numpy(),vs=vs.numpy(),expected=expected,expected_changed=expected_changed)
  rows.append({'file':file,'dtype':dtype_name,'q_len':qlen,'contexts':lengths})
(root/'fixtures.json').write_text(json.dumps({'producer_torch':torch.__version__,'scope':'CPU-generated exact-stored-value fixtures and independent FP64 reference','cases':rows},indent=2)+'\n')
print('12 CPU numerical fixtures generated; no GPU access')

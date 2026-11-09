import ast,json
from pathlib import Path
import numpy as np
import torch

root=Path('/out');width=128;chunk=64
source=ast.parse(Path('/repo/benchmarks/bin/check_gdn_extreme_decay.py').read_text())
probe=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='probe')
cases=ast.literal_eval(next(n.value for n in probe.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='cases' for t in n.targets)))
rows=[]
for heads in (48,24):
 torch.manual_seed(14929)
 query_heads=heads//3
 for index,(tokens,decay,amplitude,lengths,mixed) in enumerate(cases):
  q=torch.nn.functional.normalize(torch.randn(tokens,query_heads,width),dim=-1).bfloat16()
  k=torch.nn.functional.normalize(torch.randn(tokens,query_heads,width),dim=-1).bfloat16()
  v=torch.randn(tokens,heads,width,dtype=torch.bfloat16)*amplitude
  steps=torch.full((tokens,heads),-decay,dtype=torch.float32)
  if mixed:steps[:,1:-1]=-.02
  beta=torch.full_like(steps,.5)
  boundaries=[0]
  for length in lengths:boundaries.append(boundaries[-1]+length)
  cumulative=torch.cat([steps[i:min(i+chunk,end)].cumsum(0)
      for begin,end in zip(boundaries[:-1],boundaries[1:],strict=True)
      for i in range(begin,end,chunk)],dim=0).contiguous()
  initial=torch.randn(len(lengths),heads,width,width)*.01
  # Defining recurrence only: no chunk factorization or native intermediates.
  qr=q.double().repeat_interleave(heads//query_heads,dim=1)
  kr=k.double().repeat_interleave(heads//query_heads,dim=1)
  vr,br,gr=v.double(),beta.double(),steps.double().exp()
  expected=torch.empty_like(vr);final=torch.empty_like(initial,dtype=torch.float64)
  for sequence,(begin,end) in enumerate(zip(boundaries[:-1],boundaries[1:],strict=True)):
   state=initial[sequence].double().clone()
   for pos in range(begin,end):
    state*=gr[pos,:,None,None]
    residual=vr[pos]-(state*kr[pos,:,None,:]).sum(-1)
    state+=(br[pos,:,None]*residual)[:,:,None]*kr[pos,:,None,:]
    expected[pos]=(state*qr[pos,:,None,:]).sum(-1)*width**-.5
   final[sequence]=state
  name=f'h{heads}-case{index}.npz'
  np.savez(root/name,q=q.view(torch.uint16).numpy(),k=k.view(torch.uint16).numpy(),
      v=v.view(torch.uint16).numpy(),g=cumulative.numpy(),beta=beta.numpy(),h0=initial.numpy(),
      cu=np.array(boundaries,dtype=np.int32),expected=expected.numpy(),expected_state=final.numpy())
  rows.append({'file':name,'heads':heads,'query_heads':query_heads,'tokens':tokens,
      'decay':decay,'amplitude':amplitude,'lengths':lengths,'mixed':mixed})
  print(name,'CPU FP64 reference prepared',flush=True)
(root/'fixtures.json').write_text(json.dumps({'scope':'CPU-only independent FP64 recurrence, exact original extreme-decay cases, both head geometries','producer_torch':torch.__version__,'cases':rows},indent=2)+'\n')

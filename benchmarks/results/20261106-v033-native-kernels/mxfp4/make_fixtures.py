import json
from pathlib import Path
import numpy as np
import torch
root=Path('/out');torch.manual_seed(3308)
levels=torch.tensor([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6],dtype=torch.float64)
rows=[]
for index,(m,n,k) in enumerate([(1,128,256),(2,144,192),(17,48,256),(64,128,256),(65,128,256),(129,144,192),(2048,128,256),(1,34816,256)]):
 x=torch.randn((m,k),dtype=torch.bfloat16)
 scales=x.abs().amax(1,keepdim=True).float().clamp_min(1e-12)/448.
 a=(x.float()/scales).clamp(-448,448).to(torch.float8_e4m3fn)
 codes=torch.randint(0,16,(n,k),dtype=torch.uint8)
 w=codes[:,::2]|(codes[:,1::2]<<4)
 ws=torch.randint(119,128,(k//32,n),dtype=torch.uint8)
 wref=ws.max(0).values
 weights=levels[codes.long()]*torch.pow(2.,ws.double()-127).T.repeat_interleave(32,dim=1)
 expected=(a.double()*scales.double())@weights.T
 name=f'case{index}-m{m}-n{n}-k{k}.npz'
 np.savez(root/name,a=a.view(torch.uint8).numpy(),w=w.numpy(),ws=ws.numpy(),wref=wref.numpy(),ascale=scales.flatten().numpy(),expected=expected.numpy())
 rows.append({'file':name,'m':m,'n':n,'k':k})
 print(name,'CPU FP64 stored-operand reference ready',flush=True)
(root/'fixtures.json').write_text(json.dumps({'scope':'CPU independent FP64 dot product over exact FP8 activation/E2M1 weight/E8M0 scales','cases':rows,'producer_torch':torch.__version__},indent=2)+'\n')

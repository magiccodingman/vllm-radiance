"""Explicit diagnostic RPCs; aggregate gate spans only, no text/tensors retained."""
from radiance_platform_probe import PlatformProbe
class SpanReferenceProbe(PlatformProbe):
 def install_gdn_span_probe(self):
  import torch,radiance_gdn
  if getattr(radiance_gdn,'_qualification_span_probe',False):return {'installed':True,'rank':self.rank}
  original=radiance_gdn.fused_prefill
  stats={'calls':0,'capture_calls_skipped':0,'chunks':0,'heads_over_128':0,'heads_over_160':0,'maximum_span':0.0,'maximum_output_relative_error':0.0,'maximum_state_relative_error':0.0,'oracle_calls':0,'nonfinite_calls':0}
  def wrapped(*args,**kwargs):
   result=original(*args,**kwargs)
   if torch.cuda.is_current_stream_capturing():
    stats['capture_calls_skipped']+=1;return result
   if result is None:return result
   q,k,v,_,g,beta,scale,initial,_,cu=args[:10]
   stats['calls']+=1
   bounds=cu.cpu().tolist();data=g[0].detach().cpu().numpy()
   over=False;affected=set()
   for sequence,(begin,end) in enumerate(zip(bounds[:-1],bounds[1:])):
    for first in range(begin,end,radiance_gdn.CHUNK):
     last=min(first+radiance_gdn.CHUNK,end)-1
     spans=abs(data[last]-data[first])
     for head in (spans>128).nonzero()[0]:affected.add((sequence,int(head)))
     stats['chunks']+=1
     stats['heads_over_128']+=int((spans>128).sum());stats['heads_over_160']+=int((spans>160).sum())
     stats['maximum_span']=max(stats['maximum_span'],float(spans.max()))
     over=over or bool((spans>128).any())
   # Short fixed prompts only: independent recurrence evaluated on actual inputs.
   if over and q.shape[1]<=256:
    h=v.shape[2];hq=q.shape[2]
    qr=q[0].detach().cpu().double().repeat_interleave(h//hq,dim=1);kr=k[0].detach().cpu().double().repeat_interleave(h//hq,dim=1)
    vr=v[0].detach().cpu().double();br=beta[0].detach().cpu().double();gr=g[0].detach().cpu().double()
    expected=torch.empty_like(vr);expected_final=torch.empty_like(initial,device="cpu",dtype=torch.float64)
    for sequence,(begin,end) in enumerate(zip(bounds[:-1],bounds[1:])):
     state=initial[sequence].detach().cpu().double().clone()
     for pos in range(begin,end):
      step=gr[pos] if (pos-begin)%radiance_gdn.CHUNK==0 else gr[pos]-gr[pos-1]
      state*=step.exp()[:,None,None]
      residual=vr[pos]-(state*kr[pos,:,None,:]).sum(-1)
      state+=(br[pos,:,None]*residual)[:,:,None]*kr[pos,:,None,:]
      expected[pos]=(state*qr[pos,:,None,:]).sum(-1)*scale
     expected_final[sequence]=state
    actual,final=result
    output_error=float((actual[0].detach().cpu().double()-expected).norm()/expected.norm().clamp_min(1e-30))
    state_error=float((final.detach().cpu().double()-expected_final).norm()/expected_final.norm().clamp_min(1e-30))
    import math
    if not math.isfinite(output_error+state_error):stats['nonfinite_calls']+=1
    else:
     stats['maximum_output_relative_error']=max(stats['maximum_output_relative_error'],output_error)
     stats['maximum_state_relative_error']=max(stats['maximum_state_relative_error'],state_error)
    stats['oracle_calls']+=1
    if getattr(radiance_gdn,'_qualification_return_reference',False):
     for sequence,head in affected:
      begin,end=bounds[sequence:sequence+2]
      actual[0,begin:end,head,:].copy_(expected[begin:end,head,:].to(device=actual.device,dtype=actual.dtype))
      final[sequence,head].copy_(expected_final[sequence,head].to(device=final.device,dtype=final.dtype))
     stats['reference_head_pairs']=stats.get('reference_head_pairs',0)+len(affected)
   return result
  radiance_gdn.fused_prefill=wrapped; radiance_gdn._qualification_span_probe=True
  radiance_gdn._qualification_span_stats=stats
  return {'installed':True,'rank':self.rank}
 def gdn_span_snapshot(self):
  import radiance_gdn
  return {'rank':self.rank,**getattr(radiance_gdn,'_qualification_span_stats',{})}

 def use_control_scan(self):
  import ctypes,radiance_gdn
  library=ctypes.CDLL('/probe/control-r4d.so')
  function=library.r4d_gdn_chunk_scan_k128_v128_c64_bf16
  function.argtypes=([ctypes.c_void_p]*10+[ctypes.c_int]*6+[ctypes.c_float,ctypes.c_void_p])
  function.restype=ctypes.c_int
  radiance_gdn._qualification_control_library=library
  radiance_gdn._CHUNK_SCAN=function
  return {'control_scan':True,'rank':self.rank}

 def use_reference_prefill(self):
  import radiance_gdn
  radiance_gdn._qualification_return_reference=True
  return {'independent_fp64_prefill_reference':True,'rank':self.rank}

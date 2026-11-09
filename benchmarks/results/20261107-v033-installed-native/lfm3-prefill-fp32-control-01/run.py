import importlib.util,json,pathlib,subprocess,torch
subprocess.run(['python','/work/patch_lfm3_backport.py'],check=True)
p=pathlib.Path('/opt/vllm/lib/python3.12/site-packages/vllm/model_executor/layers/mamba/ops/causal_conv1d.py')
s=p.read_text();before='            acc += matrix_x * matrix_w  # [BLOCK_N]\n'
after=('            if KERNEL_WIDTH == 7:\n'
       '                acc += matrix_x.to(tl.float32) * matrix_w.to(tl.float32)\n'
       '            else:\n'+before.replace('            acc','                acc'))
assert s.count(before)==2
p.write_text(s.replace(before,after))
print('DIAGNOSTIC ONLY: explicit FP32 multiplication for width7; unchanged upstream assertions',flush=True)
assert torch.cuda.device_count()==1
spec=importlib.util.spec_from_file_location('lfm_tests','/work/upstream/vllm-lfm3/test_causal_conv1d.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
reference=module.causal_conv1d_ref;kernel=module.causal_conv1d_fn
records=[];captured={}
def kernel_capture(*args,**kw):
 out=kernel(*args,**kw);captured['native']=out.detach().clone();return out
def reference_capture(x,weight,bias=None,initial_states=None,return_final_states=False,final_states_out=None,activation='silu'):
 width=weight.shape[-1];length=x.shape[-1]
 extended=torch.cat([initial_states.double(),x.double()],dim=-1) if initial_states is not None else torch.nn.functional.pad(x.double(),(width-1,0))
 value=torch.zeros_like(x,dtype=torch.float64)
 for tap in range(width):value+=extended[...,tap:tap+length]*weight[:,tap].double()[None,:,None]
 if bias is not None:value+=bias.double()[None,:,None]
 exact=value if activation is None else torch.nn.functional.silu(value)
 rounded_input=value.to(x.dtype)
 two_round=rounded_input if activation is None else torch.nn.functional.silu(rounded_input)
 original=reference(x,weight,bias,initial_states,return_final_states,final_states_out,activation)
 captured.setdefault('exact',[]).append(exact);captured.setdefault('two_round',[]).append(two_round);captured.setdefault('reference',[]).append(original[0].detach().clone())
 return original
module.causal_conv1d_fn=kernel_capture;module.causal_conv1d_ref=reference_capture
for batch in (4,10):
 captured.clear();failed=False
 try:module.test_causal_conv1d_varlen(batch,False,4096,4096,7,True,True,torch.bfloat16)
 except AssertionError:failed=True
 native=captured['native'].unsqueeze(0).double();exact=torch.cat(captured['exact'],dim=-1);ref=torch.cat(captured['reference'],dim=-1).double();two=torch.cat(captured['two_round'],dim=-1).double()
 def stats(value,expected):
  error=(value-expected).abs();tolerance=0.05+0.01*expected.abs();return {'max_abs':error.max().item(),'relative_l2':((value-expected).norm()/expected.norm()).item(),'violations':int((error>tolerance).sum().item())}
 row={'batch':batch,'upstream_assertion_failed':failed,'native_vs_fp64':stats(native,exact),'reference_vs_fp64':stats(ref,exact),'reference_vs_two_round_fp64':stats(ref,two),'native_vs_reference':stats(native,ref)}
 records.append(row);print(json.dumps(row),flush=True)
pathlib.Path('/evidence/result.json').write_text(json.dumps(records,indent=2)+'\n')

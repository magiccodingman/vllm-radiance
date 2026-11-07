import pathlib,subprocess
subprocess.run(['python','/work/patch_lfm3_backport.py'],check=True)
p=pathlib.Path('/opt/vllm/lib/python3.12/site-packages/vllm/model_executor/layers/mamba/ops/causal_conv1d.py');s=p.read_text();assert s.count('_RADIANCE_CONV1D_BLOCKN = 1024')==1;p.write_text(s.replace('_RADIANCE_CONV1D_BLOCKN = 1024','_RADIANCE_CONV1D_BLOCKN = 256'))
print('DIAGNOSTIC ONLY: stock prefill channel block256; upstream tolerances unchanged',flush=True)
nodes=['upstream/vllm-lfm3/test_causal_conv1d.py::test_causal_conv1d_varlen['+str(batch)+'-False-4096-4096-7-True-True-itype0]' for batch in (4,10)]
raise SystemExit(subprocess.call(['python','-m','pytest','--noconftest',*nodes,'-q']))

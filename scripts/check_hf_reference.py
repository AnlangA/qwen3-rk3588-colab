"""Offline HF reference check for observed arithmetic/sorting discrepancies."""
import json
from pathlib import Path
import re
import subprocess
import sys
import time

def main():
    import torch
    from safetensors import safe_open
    # This is a text-only reference. Colab's preinstalled optional vision/audio
    # extensions may target a different Torch ABI from the pinned converter.
    # Disable their discovery only in this diagnostic process; no package changes.
    import transformers.utils.import_utils as optional_dependencies
    optional_dependencies._torchvision_available = False
    optional_dependencies._torchaudio_available = False
    optional_dependencies._tf_available = False
    optional_dependencies._flax_available = False
    from transformers import AutoModelForCausalLM, AutoTokenizer
    root=Path('/content/qwen3-rk3588-work')
    model_dir=root/'hf-model'
    torch.set_num_threads(2)
    index=json.loads((model_dir/'model.safetensors.index.json').read_text())['weight_map']
    inp='model.embed_tokens.weight'
    out='lm_head.weight'
    report={'source':'official local safetensors','dtype':'bfloat16','seed':42,'temperature':0,
            'scope':'two diagnostic prompts; no board execution','outputs':[]}
    with safe_open(model_dir/index[inp],framework='pt',device='cpu') as a, safe_open(model_dir/index[out],framework='pt',device='cpu') as b:
        x=a.get_tensor(inp); y=b.get_tensor(out)
        report['source_embedding_output_equal']=bool(torch.equal(x.view(torch.int16),y.view(torch.int16)))
        report['embedding_shape']=list(x.shape)
        del x,y
    print(json.dumps(report),flush=True)
    tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
    prompts=[('arithmetic','商店有24支笔，卖出9支，又进货6支。现在有多少支？只回答数字。','21'),
             ('ordering','将数字18、3、11、7按从小到大排列，只输出逗号分隔的数字。','3,7,11,18')]
    prepared=[]
    sys.path.insert(0,str(root/'llama.cpp/gguf-py'))
    from gguf import GGUFReader
    reader=GGUFReader(root/'Qwen3-1.7B-BF16.gguf')
    field=reader.fields.get('tokenizer.ggml.add_bos_token')
    report['gguf_add_bos_token']=bool(field.contents()) if field else None
    del reader
    for label,prompt,expected in prompts:
        rendered=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        inputs=tokenizer(rendered,return_tensors='pt')
        prompt_path=root/'validation_inputs'/f'{label}.txt'
        assert prompt_path.read_text()==rendered
        native=subprocess.run([str(root/'build-native/bin/llama-completion'),'-m',str(root/'Qwen3-1.7B-BF16.gguf'),
            '-ngl','0','-c','1024','--no-conversation','--no-display-prompt','--simple-io','--color','off',
            '--no-escape','--binary-file',str(prompt_path),'--verbose-prompt','--no-warmup','-n','0','-t','2'],capture_output=True,text=True,check=True,timeout=180)
        count=re.search(r'number of tokens in prompt = (\d+)',native.stderr)
        ids=[int(x) for x in re.findall(r"(?m)^(?:[0-9.]+ [A-Z] )?[ \t]*(\d+) -> '",native.stderr)]
        assert count and len(ids)==int(count.group(1)), 'Native token diagnostic incomplete'
        row={'id':label,'expected':expected,'hf_input_ids':inputs['input_ids'][0].tolist(),
             'llama_input_ids':ids,'tokenization_matches':ids==inputs['input_ids'][0].tolist()}
        prepared.append((row,inputs))
    model=AutoModelForCausalLM.from_pretrained(model_dir,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager').eval()
    report['hf_runtime_tied']=model.get_input_embeddings().weight.data_ptr()==model.get_output_embeddings().weight.data_ptr()
    for row,inputs in prepared:
        torch.manual_seed(42)
        started=time.monotonic()
        with torch.inference_mode():
            generated=model.generate(**inputs,do_sample=False,max_new_tokens=32)
        answer=tokenizer.decode(generated[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
        row.update(output=answer,elapsed_s=round(time.monotonic()-started,3))
        report['outputs'].append(row)
        print(json.dumps(row,ensure_ascii=False),flush=True)
    report['status']='completed'
    (root/'artifacts/hf_reference_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__': main()

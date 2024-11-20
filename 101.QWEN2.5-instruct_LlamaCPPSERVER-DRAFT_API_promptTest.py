# Chat with an intelligent assistant in your terminal  
# DRAFT MODEL: Qwen2.5-0.5B-Instruct-Q8_0.gguf
# INFERENCE model: Qwen2.5-7B-Instruct-Q4_0.gguf
##########################################################################
import sys
from time import sleep
import warnings
warnings.filterwarnings(action='ignore')
import datetime
from promptLibv2DRAFT import countTokens, writehistory, createCatalog
from promptLibv2DRAFT import genRANstring, createStats
from openai import OpenAI
## PREPARING FINAL DATASET

pd_id = []
pd_task = []
pd_speed = []
pd_infspeed = []
pd_ttft = []
pd_time = []
 
####################INITIALIZE THE MODEL###################################
STOPS=['<|im_end|>']
tasks = createCatalog()
NCTX = 8196
modelname = 'Qwen2.5-7B-Instruct-Q4_0.gguf'
# create THE LOG FILE 
coded5 = genRANstring(5)
logfile = f'Qwen2.5-7B-Instruct_CPPSERV-SPECDECODE_{coded5}_log.txt'
csvfile = f'Qwen2.5-7B-Instruct_CPPSERV-SPECDECODE_{coded5}.csv'
logfilename = logfile
#Write in the history the first 2 sessions
writehistory(logfilename,f'''{str(datetime.datetime.now())}
---
Your own LocalGPT with 💻 {modelname}
---
🧠🫡: You are a helpful assistant.
temperature: 0.15
repeat penalty: 1.31
max tokens: 1500
---''')    
writehistory(logfilename,f'💻: How can I assist you today in writing?')
# LOAD THE MODEL
print("\033[95;3;6m")
print("1. Waiting 10 seconds for the API to load...")
llm = OpenAI(base_url="http://localhost:8001/v1", api_key="not-needed", organization=modelname)
print(f"2. Model {modelname} loaded with LlamaCPP...")
print("\033[0m")  #reset all
history = []
print("\033[92;1m")
print(f'📝Logfile: {logfilename}')
##################### ALIGNMENT FIRST GENERATION ##############################################
question = 'Explain the plot of Cinderella in one sentence.'
test = [
    {"role": "user", "content": question}
]
print('Question:', question)
start = datetime.datetime.now()
print("💻 > ", end="", flush=True)
full_response = ""
fisrtround = 0
completion = llm.chat.completions.create(
    model="local-model", # this field is currently unused
    messages=test,
    temperature=0.25,
    frequency_penalty  = 1.53,
    max_tokens = 1200,
    stream=True,
    stop=STOPS
)
for chunk in completion:
    if chunk.choices[0].delta.content:
        if fisrtround==0:   
            print(chunk.choices[0].delta.content, end="", flush=True)
            full_response += chunk.choices[0].delta.content
            ttftoken = datetime.datetime.now() - start  
            fisrtround = 1
        else:
            print(chunk.choices[0].delta.content, end="", flush=True)
            full_response += chunk.choices[0].delta.content                      
delta = datetime.datetime.now() - start
output = full_response
print('')
print("\033[91;1m")
rating = input('Rate from 0 (BAD) to 5 (VERY GOOD) the quality of generation> ')
print("\033[92;1m")
stats = createStats(delta,question,output,rating,logfilename,'Alignment Generation',ttftoken)
print(stats)
writehistory(logfilename,f'''👨‍💻 . {question}
💻 > {output}
{stats}
''')

############################# AUTOMATIC PROMPTING EVALUATION  11 TURNS #################################
id =1
for items in tasks:
    fisrtround = 0
    task = items["task"]
    prompt = items["prompt"]
    test = []
    print(f'NLP TAKS>>> {task}')
    print("\033[91;1m")  #red
    print(prompt)
    test.append({"role": "user", "content": prompt})
    print("\033[92;1m")
    full_response = ""
    start = datetime.datetime.now()
    print("💻 > ", end="", flush=True)
    completion = llm.chat.completions.create(
        model="local-model", # this field is currently unused
        messages=test,
        temperature=0.25,
        frequency_penalty  = 1.53,
        max_tokens = 1200,
        stream=True,
        stop=STOPS
    )
    for chunk in completion:
        if chunk.choices[0].delta.content:
            if fisrtround==0:   
                print(chunk.choices[0].delta.content, end="", flush=True)
                full_response += chunk.choices[0].delta.content
                ttftoken = datetime.datetime.now() - start  
                fisrtround = 1
            else:
                print(chunk.choices[0].delta.content, end="", flush=True)
                full_response += chunk.choices[0].delta.content          
    delta = datetime.datetime.now() - start
    print('')
    print("\033[91;1m")
    rating = 'All good here only speed test with speculative decoding'#input('Rate from 0 (BAD) to 5 (VERY GOOD) the quality of generation> ')
    print("\033[92;1m")
    stats,totalseconds,speed,genspeed,ttofseconds = createStats(delta,prompt,full_response,rating,logfilename,task,ttftoken)
    print(stats)
    writehistory(logfilename,f'''👨‍💻 > {prompt}
💻 > {full_response}
{stats}
''')
    pd_id.append(id)
    pd_task.append(task)
    pd_speed.append(f'{genspeed:.2f}')
    pd_infspeed.append(f'{speed:.2f}')
    pd_ttft.append(f'{ttofseconds:.2f}')
    pd_time.append(f'{totalseconds:.2f}')   
    id += 1
# create dataframe and save to csv
zipped = list(zip(pd_id,pd_task,pd_speed,pd_infspeed,pd_ttft,pd_time))
import pandas as pdd
df = pdd.DataFrame(zipped, columns=['#', 'TASK', 'GENSPEED','INFSPEED','TTFT','TIME'])
#saving the DataFrame as a CSV file 
df_csv_data = df.to_csv(csvfile, index = False, encoding='utf-8') 
print('\nCSV String:\n', df)     
from rich.console import Console
console = Console()
console.print('---')
console.print(df)   



##############MODEL CARD##########################################
"""
# Chat with an intelligent assistant in your terminal  
# MODEL: https://huggingface.co/bartowski/AMD-OLMo-1B-SFT-DPO-GGUF
# Original model: https://huggingface.co/amd/AMD-OLMo-1B-SFT-DPO
# AMD-OLMo-1B-SFT-DPO-Q6_K_L.gguf
MODELCARD
===========================================================
mp = 'models/AMD-OLMo-1B-SFT-DPO-Q6_K_L.gguf'
from llama_cpp import Llama
llm = Llama(model_path = mp)

CHAT TEMPLATE = yes
NCTX = 2048

Prompt format
```
<|system|>
{system_prompt}
<|user|>
{prompt}
<|assistant|>
```

Available chat formats from metadata: chat_template.default
Using gguf chat template: {% for message in messages %}
{% if message['role'] == 'user' %}
{{ '<|user|>
' + message['content'] }}
{% elif message['role'] == 'system' %}
{{ '<|system|>
' + message['content'] }}
{% elif message['role'] == 'assistant' %}
{{ '<|assistant|>
'  + message['content'] }}
{% endif %}
{% if loop.last and add_generation_prompt %}
{{ '<|assistant|>' }}
{% endif %}
{% endfor %}
Using chat eos_token: |||IP_ADDRESS|||
Using chat bos_token: |||IP_ADDRESS|||


llama_model_loader: loaded meta data with 33 key-value pairs and 113 tensors from models/AMD-OLMo-1B-SFT-DPO-Q6_K_L.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
             general.architecture str              = olmo
                     general.type str              = model
                     general.name str              = AMD OLMo 1B SFT DPO
                 general.finetune str              = SFT-DPO
                 general.basename str              = AMD-OLMo
               general.size_label str              = 1B
                  general.license str              = apache-2.0
                 general.datasets arr[str,1]       = ["allenai/dolma"]
                 olmo.block_count u32              = 16
              olmo.context_length u32              = 2048
            olmo.embedding_length u32              = 2048
         olmo.feed_forward_length u32              = 8192
        olmo.attention.head_count u32              = 16
     olmo.attention.head_count_kv u32              = 16
              olmo.rope.freq_base f32              = 10000.000000
                general.file_type u32              = 18
olmo.attention.layer_norm_epsilon f32              = 0.000010
             tokenizer.ggml.model str              = gpt2
               tokenizer.ggml.pre str              = olmo
            tokenizer.ggml.tokens arr[str,50304]   = ["<|endoftext|>", "<|padding|>", "!",...
        tokenizer.ggml.token_type arr[i32,50304]   = [3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
            tokenizer.ggml.merges arr[str,50009]   = ["Ġ Ġ", "Ġ t", "Ġ a", "h e", "i n...
      tokenizer.ggml.bos_token_id u32              = 50279
      tokenizer.ggml.eos_token_id u32              = 50279
  tokenizer.ggml.padding_token_id u32              = 1
     tokenizer.ggml.add_bos_token bool             = false
     tokenizer.ggml.add_eos_token bool             = false
          tokenizer.chat_template str              = {% for message in messages %}\n{% if m...
     general.quantization_version u32              = 2
            quantize.imatrix.file str              = /models_out/AMD-OLMo-1B-SFT-DPO-GGUF/...
         quantize.imatrix.dataset str              = /training_dir/calibration_datav3.txt
   quantize.imatrix.entries_count i32              = 112
    quantize.imatrix.chunks_count i32              = 132
llama_model_loader: - type q8_0:    1 tensors
llama_model_loader: - type q6_K:  112 tensors
llm_load_vocab: special tokens cache size = 28
llm_load_vocab: token to piece cache size = 0.2985 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = olmo
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 50304
llm_load_print_meta: n_merges         = 50009
llm_load_print_meta: vocab_only       = 0
llm_load_print_meta: n_ctx_train      = 2048
llm_load_print_meta: n_embd           = 2048
llm_load_print_meta: n_layer          = 16
llm_load_print_meta: n_head           = 16
llm_load_print_meta: n_head_kv        = 16
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_swa            = 0
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: n_embd_k_gqa     = 2048
llm_load_print_meta: n_embd_v_gqa     = 2048
llm_load_print_meta: f_norm_eps       = 1.0e-05
llm_load_print_meta: f_norm_rms_eps   = 0.0e+00
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 8192
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 2048
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: ssm_dt_b_c_rms   = 0
llm_load_print_meta: model type       = ?B
llm_load_print_meta: model ftype      = Q6_K
llm_load_print_meta: model params     = 1.18 B
llm_load_print_meta: model size       = 944.39 MiB (6.73 BPW)
llm_load_print_meta: general.name     = AMD OLMo 1B SFT DPO
llm_load_print_meta: BOS token        = 50279 '|||IP_ADDRESS|||'
llm_load_print_meta: EOS token        = 50279 '|||IP_ADDRESS|||'
llm_load_print_meta: PAD token        = 1 '<|padding|>'
llm_load_print_meta: LF token         = 128 'Ä'
llm_load_print_meta: EOT token        = 0 '<|endoftext|>'
llm_load_print_meta: max token length = 1024
"""
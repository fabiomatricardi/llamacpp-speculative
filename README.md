# llamacpp-speculative decoding
Comparison of using llama.cpp llama-cpp-python server with speculative decoding and without it

> There is really no benefit?

In my findings, without offloading any layer oì to GPU, there are no increase in speed.

##  Comparison table
<img src='https://github.com/fabiomatricardi/llamacpp-speculative/raw/main/TableComparison.png' width=900>



## STACK
- llama-cpp-python[server] revision 0.3.2 that supports also Granite3, OLMO and MoE
- openAI pytho library to access the endpoints
- automatic prompt evaluation
- server settings as per [documentation](https://llama-cpp-python.readthedocs.io/en/latest/server/)

## Dependencies
- llama-cpp-python[server]==0.3.2
- openai
- tiktoken


## How to use
### with speculative decoding
from one terminal window run
```
python -m llama_cpp.server --config_file .\QWENsettingsDRAFT.me
```
Using the following server settings
```
{
    "port" : 8001,
    "models": [
        {
            "model": "models/Qwen2.5-7B-Instruct-Q4_0.gguf",
            "model_alias": "QWEN2.5",
            "n_ctx": 8196,
            "draft_model" : "models/Qwen2.5-0.5B-Instruct-Q8_0.gguf",
            "draft_model_num_pred_tokens" : 2
        }
    ]
}
```
In another terminal window run
```
python .\101.QWEN2.5-instruct_LlamaCPPSERVER-DRAFT_API_promptTest.py
```

### WITHOUT speculative decoding
from one terminal window run
```
python -m llama_cpp.server --config_file .\QWENsettingsNODRAFT.me
```
Using the following server settings
```
{
    "port" : 8001,
    "models": [
        {
            "model": "models/Qwen2.5-7B-Instruct-Q4_0.gguf",
            "model_alias": "QWEN2.5",
            "n_ctx": 8196
        }
    ]
}
```
In another terminal window run
```
python .\102.QWEN2.5-instruct_LlamaCPPSERVER-NODRAFT_API_promptTest.py
```



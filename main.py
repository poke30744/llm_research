import typer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from threading import Thread
from transformers import TextIteratorStreamer

def main(model_name: str="Qwen/Qwen2-1.5B-Instruct",
         system_prompt: str="You are a helpful assistant."
         ):
    
    MAX_INPUT_TOKEN_LENGTH = 131072

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
       model_name,
        torch_dtype=torch.float16,
        device_map="auto")
    
    #chat_history = []
    conversation = []
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})
    
    while True:
        message: str = input("Ready: ")
        conversation.append({"role": "user", "content": message})

        input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt")
        if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
            input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
            print(f"Trimmed input from conversation as it was longer than {MAX_INPUT_TOKEN_LENGTH} tokens.")
        input_ids = input_ids.to(model.device)

        streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            {"input_ids": input_ids},
            streamer=streamer,
            max_new_tokens=1024,
            do_sample=True,
            top_p=0.8,
            top_k=10,
            temperature=0.2,
            num_beams=1,
            repetition_penalty=1.2,
        )

        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()

        try:
            outputs = []
            for text in streamer:
                outputs.append(text)
                print(text, end="", flush=True)
        finally:
            t.join()
        print()
        conversation.append({"role": "assistant", "content": ''.join(outputs)})

if __name__ == "__main__":
    typer.run(main)
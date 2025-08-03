"""
coding   : utf-8
@Date    : 2024/7/10
@Author  : Shaobo
@Describe: 
"""

import torch
from protocols.openai_api import ChatCompletionRequest, ChatCompletionStreamResponse, ChatCompletionResponse
from sseclient import Event
from transformers import AutoTokenizer, AutoModel


class CodegeexChatModel:
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        if args.bf16:
            self.model = AutoModel.from_pretrained(
                args.model_name_or_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            ).to(args.device).eval()
        else:
            self.model = AutoModel.from_pretrained(
                args.model_name_or_path,
                trust_remote_code=True
            ).to(args.device).eval()
        print("Model is initialized.")

    def stream_chat(self, request: ChatCompletionRequest):
        try:
            inputs = self.tokenizer.apply_chat_template(
                conversation=[msg.model_dump() for msg in request.messages],
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True
            ).to(self.model.device)
            gen_configs = {
                "max_new_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "repetition_penalty": request.presence_penalty,
                "do_sample": True if request.temperature else request.temperature,
            }
            length = 0
            for outputs in self.model.stream_generate(**inputs, **gen_configs):
                response = self.tokenizer.decode(outputs.tolist()[0][len(inputs["input_ids"][0]):-1])
                if not response or response[-1] == "�":
                    continue
                resp = ChatCompletionStreamResponse()
                resp.choices[0].delta.content = response[length:]
                event = Event(data=resp.json(), event='message')
                yield event.dump()
                length = len(response)
            resp = ChatCompletionStreamResponse()
            resp.choices[0].finish_reason = 'stop'
            event = Event(data=resp.json(), event='message')
            yield event.dump()
        except Exception as e:
            resp = ChatCompletionStreamResponse()
            resp.choices[0].finish_reason = 'stop'
            event = Event(data=f"请求报错，错误原因：{e}", event='message')
            yield event.dump()
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def chat(self, request: ChatCompletionRequest):
        try:
            response, _ = self.model.chat(
                self.tokenizer,
                query=request.messages[-1].content,
                history=[msg.model_dump() for msg in request.messages[:-1]],
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                repetition_penalty=request.presence_penalty
            )
            if isinstance(response, dict):
                response = '\n'.join([response.get('name'), response.get('content') or str(response.get('parameters'))])
            resp = ChatCompletionResponse()
            resp.choices[0].message.content = response
            resp.choices[0].finish_reason = 'stop'
            return resp.model_dump()
        except Exception as e:
            return f"请求报错，错误原因：{e}"
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

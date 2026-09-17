import re

from transformers import GenerationConfig, pipeline


class LLM:

    def __init__(self) -> None:
        self.pipe = pipeline("text-generation", model='Qwen/Qwen3-0.6B',
                             device_map='auto',
                             clean_up_tokenization_spaces=False)

    def _generate_answer(self, question: str, context: str) -> str:
        message = [
            {'role': 'system',
                'content': f"""Answer the user's question using ONLY the
                provided context.

Rules:
- The context is : {context}.
- '{question}' is never a part of the context.
- Do NOT use external knowledge.
- Do NOT invent information.
- If the context does not contain enough information do NOT answer. Just
    say that the answer cannot be determined from the provided context.
- Be concise and answer the question using technical terms from the context."""
            },
            {'role': 'user',
                'content': f'{question} /no_think'}
        ]
        gen_config = GenerationConfig.from_pretrained('Qwen/Qwen3-0.6B')
        gen_config.max_new_tokens = 256
        output = self.pipe(message, generation_config=gen_config)
        llm_response = str(output[0]["generated_text"][-1]["content"])
        return re.sub(r"<think>[\s\S]*?<\/think>\s*", '', llm_response)

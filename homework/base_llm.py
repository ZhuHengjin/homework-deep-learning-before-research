from typing import overload

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class BaseLLM:
    def __init__(self, checkpoint=checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        self.device = device

    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into an input to SmolLM2. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
        return question

    def parse_answer(self, answer: str) -> float:
        """
        Parse the <answer></answer> tag and return a float.
        This function is somewhat robust to output errors (e.g. missing </answer> tags).
        """
        try:
            return float(answer.split("<answer>")[1].split("</answer>")[0])
        except (IndexError, ValueError):
            return float("nan")

    def generate(self, prompt: str) -> str:
        """
        (Optional) Implement this method first and then implement batched_generate below.
        It is much easier to implement generation without batching.

        The overall flow is the same:
        - tokenize the prompt with self.tokenizer
        - call self.model.generate
        - decode the outputs with self.tokenizer.decode

        """
        return self.batched_generate([prompt])[0]

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: None = None, temperature: float = 0
    ) -> list[str]:
        """
        Batched version of `generate` method.
        This version returns a single generation for each prompt.
        """

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: int, temperature: float = 0
    ) -> list[list[str]]:
        """
        Batched version of `generate` method.
        This version returns a list of generation for each prompt.
        """

    def batched_generate(
        self, 
        prompts: list[str], 
        num_return_sequences: int | None = None, 
        temperature: float = 0
        ) -> list[str] | list[list[str]]:
        """
        Batched version of `generate` method.

        You will likely get an up to 10x speedup using batched decoding.

        To implement batch decoding you will need to:
        - tokenize the prompts self.tokenizer with padding=True and return_tensors="pt"
        - call self.model.generate
        - decode the outputs with self.tokenizer.batch_decode

        Tip: You need to set self.tokenizer.padding_side = "left" to get the correct padding behavior for generation.
            Left padding makes sure all sequences are aligned to the right (i.e. where tokens are generated).
        Tip: self.model.generate takes a lot of parameters. Here are some relevant ones:
            - max_new_tokens: The maximum number of tokens to generate. Set this to a reasonable value
                            (50 should suffice).
            - do_sample and temperature: For any temperature > 0, set do_sample=True.
                                        do_sample=False will use greedy decoding.
            - num_return_sequences: The number of sequences to return. Note that this will generate a flat
                                    list of len(prompts) * num_return_sequences entries.
            - eos_token_id: The end of sequence token id. This is used to stop generation. Set this
                            to self.tokenizer.eos_token_id.
        Pro Tip: Only batch_decode generated tokens by masking out the inputs with
                outputs[:, len(inputs["input_ids"][0]) :]
        """
        from tqdm import tqdm  # Importing tqdm for progress bar

        # Preventing OOM
        # Depending on your GPU batched generation will use a lot of memory.
        # If you run out of memory, try to reduce the micro_batch_size.
        micro_batch_size = 32
        if len(prompts) > micro_batch_size:
            if num_return_sequences is None:
                return [
                    r
                    for idx in tqdm(
                        range(0, len(prompts), micro_batch_size), desc=f"LLM Running on Micro Batches {micro_batch_size}"
                    )
                    for r in self.batched_generate(prompts[idx : idx + micro_batch_size], num_return_sequences, temperature)
                ]
            else:
                results = []
                for idx in tqdm(
                    range(0, len(prompts), micro_batch_size), desc=f"LLM Running on Micro Batches {micro_batch_size}"
                ):
                    batch_results = self.batched_generate(
                        prompts[idx : idx + micro_batch_size], num_return_sequences, temperature
                    )
                    results.extend(batch_results)
                return results
        
        # Set padding side to left for proper alignment during generation
        self.tokenizer.padding_side = "left"
        
        # Tokenize all prompts at once with padding
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)
        
        # Set up generation parameters
        gen_kwargs = {
            "max_new_tokens": 50,  # reasonable limit for unit conversion answers
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # Handle temperature setting
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        
        # Handle multiple return sequences if requested
        if num_return_sequences is not None and num_return_sequences > 1:
            gen_kwargs["num_return_sequences"] = num_return_sequences
        
        # Generate outputs
        outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **gen_kwargs)

        
        # Decode the generated outputs
        generations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Reshape if multiple sequences per prompt were requested
        if num_return_sequences is not None and num_return_sequences > 1:
            # Reshape flat list into list of lists
            reshaped_generations = []
            for i in range(0, len(generations), num_return_sequences):
                reshaped_generations.append(generations[i:i + num_return_sequences])
            return reshaped_generations
        
        return generations

    # def batched_generate(
    #     self, prompts: list[str], num_return_sequences: int | None = None, temperature: float = 0
    # ) -> list[str] | list[list[str]]:
    #     """
    #     Batched version of `generate` method.

    #     You will likely get an up to 10x speedup using batched decoding.

    #     To implement batch decoding you will need to:
    #     - tokenize the prompts self.tokenizer with padding=True and return_tensors="pt"
    #     - call self.model.generate
    #     - decode the outputs with self.tokenizer.batch_decode

    #     Tip: You need to set self.tokenizer.padding_side = "left" to get the correct padding behavior for generation.
    #          Left padding makes sure all sequences are aligned to the right (i.e. where tokens are generated).
    #     Tip: self.model.generate takes a lot of parameters. Here are some relevant ones:
    #         - max_new_tokens: The maximum number of tokens to generate. Set this to a reasonable value
    #                           (50 should suffice).
    #         - do_sample and temperature: For any temperature > 0, set do_sample=True.
    #                                      do_sample=False will use greedy decoding.
    #         - num_return_sequences: The number of sequences to return. Note that this will generate a flat
    #                                 list of len(prompts) * num_return_sequences entries.
    #         - eos_token_id: The end of sequence token id. This is used to stop generation. Set this
    #                         to self.tokenizer.eos_token_id.
    #     Pro Tip: Only batch_decode generated tokens by masking out the inputs with
    #              outputs[:, len(inputs["input_ids"][0]) :]
    #     """
    #     from tqdm import tqdm  # Importing tqdm for progress bar

    #     # Preventing OOM
    #     # Depending on your GPU batched generation will use a lot of memory.
    #     # If you run out of memory, try to reduce the micro_batch_size.
    #     micro_batch_size = 32
    #     if len(prompts) > micro_batch_size:
    #         return [
    #             r
    #             for idx in tqdm(
    #                 range(0, len(prompts), micro_batch_size), desc=f"LLM Running on Micro Batches {micro_batch_size}"
    #             )
    #             for r in self.batched_generate(prompts[idx : idx + micro_batch_size], num_return_sequences, temperature)
    #         ]

    #     self.tokenizer.padding_side = "left"
    #     inputs = self.tokenizer(
    #         prompts,
    #         padding=True,
    #         return_tensors="pt"
    #     ).to(self.device)

    #     outputs = self.model.generate(
    #         input_ids=inputs["input_ids"],
    #         attention_mask=inputs["attention_mask"],
    #         max_new_tokens=50,
    #         do_sample=(temperature > 0),
    #         temperature=temperature,
    #         num_return_sequences=(num_return_sequences or 1),
    #         eos_token_id=self.tokenizer.eos_token_id
    #     )

    #     # Remove the prompt tokens from the output before decoding
    #     prompt_len = inputs["input_ids"].shape[1]
    #     generated_tokens = outputs[:, prompt_len:]
    #     decoded = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    #     # Handle output shape
    #     if num_return_sequences is None or num_return_sequences == 1:
    #         return decoded
    #     else:
    #         return [decoded[i:i + num_return_sequences] for i in range(0, len(decoded), num_return_sequences)]

    def answer(self, *questions) -> list[float]:
        """
        Answer questions given as individual string arguments.
        """
        # Convert each question
        prompts = [self.format_prompt(q) for q in questions]
        generations = self.batched_generate(prompts)
        return [self.parse_answer(g) for g in generations]


def test_model():
    # The following code simply tests of the BaseLLM is able to complete text.
    # It should produce garbage answers, but it should not crash.
    # In my case it talks about cats eating cats, and dogs being happy.
    testset = ["The cat went up", "The dog went down"]
    model = BaseLLM()
    for t in testset:
        print("testing generate function")
        print("input", t)
        answer = model.generate(t)
        print("output", answer)
    answers = model.batched_generate(testset)
    print(answers)


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model})

from .base_llm import BaseLLM


class CoTModel(BaseLLM):
\
    def format_prompt(self, question: str) -> str:

        system_content = "Be concise. Use the <answer></answer> tag to indicate the answer."
        example_question = "How many inches are in 6.7 kilometers?"

        messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": "How many grams are in 2.5 kilograms?"},
                    {"role": "assistant", "content": "To convert kilograms to grams, I multiply by 1000.\n2.5 kilograms x 1000 = <answer>2500</answer>"},
                    {"role": "user", "content": question},
                ]


        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})

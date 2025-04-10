import openai
import ast
import os


class OpenAIClient:
    def __init__(self, model="gpt-4-turbo", max_tokens=4000, presence_penalty=0, temperature=0.1, top_p=0.9):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable.")

        openai.api_key = self.api_key  # Setting API key securely

        self.model = model
        self.max_tokens = max_tokens
        self.presence_penalty = presence_penalty
        self.temperature = temperature
        self.top_p = top_p

    def generate_response(self, system_prompt_file: str, user_prompt: str):
        try:
            # Load system prompt from file
            with open(f"prompts/{system_prompt_file}.txt", "r", encoding="utf-8") as file:
                system_prompt = file.read().strip()

            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                presence_penalty=self.presence_penalty,
                temperature=self.temperature,
                top_p=self.top_p
            )

            sanitized_content = response.choices[0].message.content.strip()
            result = ast.literal_eval(sanitized_content.lstrip('```json').rstrip('```').strip())

            return result

        except FileNotFoundError:
            raise FileNotFoundError(f"System prompt file '{system_prompt_file}.txt' not found in 'prompts' directory.")
        except Exception as e:
            raise RuntimeError(f"Error generating response: {e}")

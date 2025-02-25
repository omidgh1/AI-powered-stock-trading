import openai
import ast

openai.api_key = ("sk-proj-2222ED7vJmFIxotBemcXr89q1b1-icWbDdGlAW2anyHjY_9koMG4WNa0OaB3-"
                  "FBUt5wpFZ_rTDT3BlbkFJXYao4vwTQniI4zykzToKxIyh89Rlfn4-gZ9nO5YWI-NexcekFIWi-_NnYLOBhwFVIOfCHk8QYA")


def AI_generator(system_prompt, user_prompt,
                 model='gpt-4-turbo',
                 max_tokens=4000,
                 presence_penalty=0,
                 temperature=0.1,
                 top_p=0.9):
    with open(f"prompts/{system_prompt}.txt", "r") as file:
        system_prompt = file.read()

    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=max_tokens,
        presence_penalty=presence_penalty,
        temperature=temperature,
        top_p=top_p
    )

    sanitized_content = response.choices[0].message.content.strip()
    result = ast.literal_eval(sanitized_content.lstrip('```json').rstrip('```').strip())

    return result

result = AI_generator(system_prompt='Influential_People',user_prompt='GOOGL')

print(result)
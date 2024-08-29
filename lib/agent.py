"""
### 使用示例
agent = TranslatorAgent(api_key="your_api_key")
result = agent.translate("en", "zh", "Hello, world!", "China")
"""
from typing import Union
import http.client
import json
import ssl
from urllib.parse import urlparse

MAX_TOKENS_PER_CHUNK = 2000

class TranslatorAgent:
    def __init__(self, api_key: str, endpoint:str = "https://api.openai.com/v1/chat/completions", model: str = "gpt-4o", max_tokens: int = MAX_TOKENS_PER_CHUNK):
        self.api_key = api_key
        self.endpoint = endpoint
        self.model = model
        self.max_tokens = max_tokens

    def get_completion(
        self,
        prompt: str,
        system_message: str = "You are a helpful assistant.",
        temperature: float = 0.3,
        json_mode: bool = False,
    ) -> Union[str, dict]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "top_p": 1,
        }

        if json_mode:
            data["response_format"] = {"type": "json_object"}

        parsed_url = urlparse(self.endpoint)
        conn = http.client.HTTPSConnection(parsed_url.netloc, context=ssl._create_unverified_context())
        conn.request("POST", parsed_url.path, body=json.dumps(data), headers=headers)
        response = conn.getresponse()
        
        if response.status != 200:
            raise Exception(f"API request failed with status {response.status}: {response.read().decode()}")
        
        result = json.loads(response.read().decode())
        conn.close()

        if json_mode:
            return result["choices"][0]["message"]["content"]
        else:
            return result["choices"][0]["message"]["content"]

    def one_chunk_initial_translation(
        self,
        source_lang: str,
        target_lang: str,
        source_text: str
    ) -> str:
        system_message = f"You are an expert linguist, specializing in translation from {source_lang} to {target_lang}."

        translation_prompt = f"""This is an {source_lang} to {target_lang} translation, please provide the {target_lang} translation for this text. \
Do not provide any explanations or text apart from the translation.
{source_lang}: {source_text}

{target_lang}:"""

        return self.get_completion(translation_prompt, system_message=system_message)

    def one_chunk_reflect_on_translation(
        self,
        source_lang: str,
        target_lang: str,
        source_text: str,
        translation_1: str,
        country: str = "",
    ) -> str:
        system_message = f"You are an expert linguist specializing in translation from {source_lang} to {target_lang}. \
You will be provided with a source text and its translation and your goal is to improve the translation."

        if country != "":
            reflection_prompt = f"""Your task is to carefully read a source text and a translation from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions to improve the translation. \
The final style and tone of the translation should match the style of {target_lang} colloquially spoken in {country}.

The source text and initial translation, delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT> and <TRANSLATION></TRANSLATION>, are as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's \n\
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),\n\
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
(iii) style (by ensuring the translations reflect the style of the source text and take into account any cultural context),\n\
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_lang}).\n\

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else."""
        else:
            reflection_prompt = f"""Your task is to carefully read a source text and a translation from {source_lang} to {target_lang}, and then give constructive criticisms and helpful suggestions to improve the translation. \

The source text and initial translation, delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT> and <TRANSLATION></TRANSLATION>, are as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's \n\
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),\n\
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
(iii) style (by ensuring the translations reflect the style of the source text and take into account any cultural context),\n\
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_lang}).\n\

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else."""
        return self.get_completion(reflection_prompt, system_message=system_message)

    def one_chunk_improve_translation(
        self,
        source_lang: str,
        target_lang: str,
        source_text: str,
        translation_1: str,
        reflection: str,
    ) -> str:
        system_message = f"You are an expert linguist, specializing in translation editing from {source_lang} to {target_lang}."

        prompt = f"""Your task is to carefully read, then edit, a translation from {source_lang} to {target_lang}, taking into
account a list of expert suggestions and constructive criticisms.

The source text, the initial translation, and the expert linguist suggestions are delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT>, <TRANSLATION></TRANSLATION> and <EXPERT_SUGGESTIONS></EXPERT_SUGGESTIONS> \
as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

<EXPERT_SUGGESTIONS>
{reflection}
</EXPERT_SUGGESTIONS>

Please take into account the expert suggestions when editing the translation. Edit the translation by ensuring:

(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules and ensuring there are no unnecessary repetitions), \
(iii) style (by ensuring the translations reflect the style of the source text)
(iv) terminology (inappropriate for context, inconsistent use), or
(v) other errors.

Output only the new translation and nothing else."""

        return self.get_completion(prompt, system_message)

    def one_chunk_translate_text(
        self,
        source_lang: str,
        target_lang: str,
        source_text: str,
        country: str = "",
    ) -> str:
        translation_1 = self.one_chunk_initial_translation(source_lang, target_lang, source_text)
        reflection = self.one_chunk_reflect_on_translation(source_lang, target_lang, source_text, translation_1, country)
        translation_2 = self.one_chunk_improve_translation(source_lang, target_lang, source_text, translation_1, reflection)
        return translation_2


    def translate(
        self,
        source_lang: str,
        target_lang: str,
        source_text: str,
        country: str,
    ) -> str:
       
        return self.one_chunk_translate_text(source_lang, target_lang, source_text, country)


if __name__ == '__main__':
    import os
    text = '''I shall be telling this with a sigh
Somewhere ages and ages hence;
Two roads diverged in a wood, and I―
I took the one less traveled by,
And that has made all the difference.'''
    agent = TranslatorAgent(api_key=os.getenv("OPENAI_API_KEY"), endpoint=os.getenv("OPENAI_API_ENDPOINT"), model=os.getenv("OPENAI_MODEL_NAME"))
    print(agent.translate("en", "zh", text, "China"))
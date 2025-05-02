import replicate
import os
import json


# API 토큰 직접 설정
os.environ["REPLICATE_API_TOKEN"] = "r8_5fTF4b4eXoWpdkwcN3MTxslFDGKFLii39eE80"
os.environ["REPLICATE_TIMEOUT"] = "600"
Best_prompt = "Generate 10 unique captions for the provided video frame, each from a different perspective"
output = replicate.run(
    "meta/meta-llama-3-8b",
    input={
        "prompt": """
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant. Answer in this JSON format: {"caption": "<rewritten_caption>"}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Rewrite the following caption to make it clearer and more detailed : "A cat is watching at the window."<|eot_id|>
""",
        "max_tokens": 200,
        "temperature": 0.6,
        "top_p": 0.9,
        # "prompt_template": "{prompt}",
        "presence_penalty": 1.15,
        "frequency_penalty": 0.2
    }
)

print(output)

full_output = "".join(output)

# JSON 파일로 저장
result = {"response": full_output}
with open("llama3_output.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=4)

print("응답을 llama3_output.json 파일로 저장했습니다.")
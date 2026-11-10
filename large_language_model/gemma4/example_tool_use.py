import ailia_llm

import json
import os
import urllib.request

model_file_path = "gemma-4-E2B-it-Q4_K_M.gguf"
if not os.path.exists(model_file_path):
    print("Model file not found. Downloading...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/ailia-models/gemma/gemma-4-E2B-it-Q4_K_M.gguf",
        model_file_path
    )

# Tool implementation (mock)
def get_weather(city):
    return {"city": city, "weather": "sunny", "temperature": 25}

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather of a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

model = ailia_llm.AiliaLLM()
model.open(model_file_path, n_ctx=4096)

try:
    model.set_tools(tools)

    messages = [{"role": "user", "content": "東京の天気を教えてください。"}]

    for turn in range(8):
        # Deltas are optional previews, the SDK accumulates the response
        for delta_text in model.generate_json(messages):
            print(delta_text, end="", flush=True)
        print("")

        try:
            response = model.get_response_json()
        except ailia_llm.AiliaLLMError as error:
            if error.code != ailia_llm.AILIA_LLM_STATUS_PARSE_ERROR:
                raise
            # Do not append the failed assistant message
            messages.append({"role": "user", "content": "Please retry with a complete response or tool call."})
            continue

        messages.append(response)

        if not response.get("tool_calls"):
            print("Answer :", response["content"])
            break

        for call in response["tool_calls"]:
            name = call["function"]["name"]
            args = json.loads(call["function"]["arguments"])
            print("Tool call :", name, args)
            if name != "get_weather":
                raise ValueError("Unknown tool " + name)
            if not isinstance(args, dict) or not isinstance(args.get("city"), str) or not args["city"]:
                raise ValueError("Invalid city argument")
            result = get_weather(args["city"])
            print("Tool result :", result)
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps(result, ensure_ascii=False)
            })
    else:
        raise RuntimeError("No final answer within 8 turns")

    if model.context_full():
        raise Exception("Context full")
finally:
    model.set_tools(None)

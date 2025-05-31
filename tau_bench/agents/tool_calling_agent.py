# Copyright Sierra

import os
import json
from litellm import completion, responses
from typing import List, Optional, Dict, Any

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME

from typing import Any, Dict, List, Mapping, Union

try:
    from openai.types.responses import Response as ResponsesAPIResponse
except ImportError:
    ResponsesAPIResponse = object


# -------- helpers ------------------------------------------------------------
def _as_dict(obj: Any) -> Mapping[str, Any]:
    """A unified mapping view over pydantic models or plain dicts."""
    if hasattr(obj, "model_dump"):      # pydantic v2
        return obj.model_dump(exclude_none=True)
    if hasattr(obj, "dict"):            # pydantic v1
        return obj.dict(exclude_none=True)
    return obj                          # already a dict


def _iter_items(resp: Union[ResponsesAPIResponse, Mapping[str, Any]]):
    """Yield every `output` item as a dict, preserving order."""
    seq = resp.output if hasattr(resp, "output") else resp.get("output", [])
    for itm in seq:
        yield _as_dict(itm)


def _collapse_text(content_parts: List[Any]) -> str:
    """Concatenate all `output_text` fragments into one string."""
    chunks: List[str] = []
    for part in content_parts:
        p = _as_dict(part)
        if p.get("type") in {"output_text", "text"}:
            chunks.append(p.get("text", ""))
    return "".join(chunks)


# -------- main converter ------------------------------------------------------
def responses_to_chat_message(
    resp: Union[ResponsesAPIResponse, Mapping[str, Any]]
) -> Dict[str, Any]:
    """
    Convert a **Responses API** Response object (or raw JSON dict)
    into the single `choices[n].message` object used by **Chat Completions**.
    """
    message: Dict[str, Any] = {"role": "assistant"}  # default scaffold
    tool_calls: List[Dict[str, Any]] = []

    for item in _iter_items(resp):
        typ = item.get("type")

        if typ == "message" and "content" in item and "content" not in message:
            # first assistant text block
            text = _collapse_text(item["content"])
            message["content"] = text if text else [_as_dict(p) for p in item["content"]]

        elif typ in {"function_call", "file_search_tool_call",
                     "function_web_search", "computer_tool_call"}:
            # unify to Chat‑Completions `tool_calls` format
            if typ == "function_call":
                tool_calls.append(
                    {
                        "id": item.get("call_id"),
                        "type": "function",
                        "function": {
                            "name": item.get("name"),
                            "arguments": item.get("arguments") or "{}",
                        },
                    }
                )
            else:          # built‑in tools keep their own `type`
                tool_calls.append(
                    {
                        "id": item.get("call_id"),
                        "type": typ.replace("_tool_call", ""),  # e.g. "file_search"
                        **{k: v for k, v in item.items() if k not in {"type", "call_id"}},
                    }
                )

    # if the model never produced a plain message, mimic Chat API behaviour
    message.setdefault("content", None)
    if tool_calls:
        message["tool_calls"] = tool_calls

    return message

def _convert_to_input_messages(m):
    if m.get('tool_calls'):
        return [tool['function'] | {"call_id": tool['id'], "type": "function_call"} for tool in m['tool_calls']]
    if m.get("role") == "tool":
        return [{"call_id": m["tool_call_id"], "output": m["content"], "type": "function_call_output"}]
    else:
        return [m]

class ToolCallingAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
    ):
        self.tools_info = tools_info
        self.wiki = wiki
        self.model = model
        self.provider = provider
        self.temperature = temperature

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        total_cost = 0.0
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        reward = 0.0
        use_responses_api =  os.environ.get("USE_RESPONSES_API")
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.wiki},
            {"role": "user", "content": obs},
        ]
        reasoning_effort = os.environ.get("REASONING_EFFORT", "medium")
        if use_responses_api:
            tools_info = [{"type": "function"} | tool["function"] for tool in self.tools_info]
        previous_response_id = None
        for _ in range(max_num_steps):
            if use_responses_api:
                messages = [
                    item
                    for m in messages
                    for item in _convert_to_input_messages(m)
                ]
                if previous_response_id:
                    input_messages = messages[-1:]
                else:
                    input_messages = messages
                res = responses(
                   input=input_messages,
                   model=self.model,
                   previous_response_id=previous_response_id,
                   custom_llm_provider=self.provider,
                   tools=tools_info,
                   store=True,
                   reasoning={"effort": reasoning_effort},
                )
                previous_response_id= res.id
                # next_message = res.choices[0].message.model_dump()
                next_message = responses_to_chat_message(res)
            else:
                res = completion(
                    messages=messages,
                    model=self.model,
                    custom_llm_provider=self.provider,
                    tools=self.tools_info,
                    temperature=self.temperature,
                    reasoning_effort=reasoning_effort,
                )
                next_message = res.choices[0].message.model_dump()

            total_cost += 0 # res._hidden_params["response_cost"]
            action = message_to_action(next_message)
            env_response = env.step(action)
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}
            if action.name != RESPOND_ACTION_NAME:
                next_message["tool_calls"] = next_message["tool_calls"][:1]
                messages.extend(
                    [
                        next_message,
                        {
                            "role": "tool",
                            "tool_call_id": next_message["tool_calls"][0]["id"],
                            "name": next_message["tool_calls"][0]["function"]["name"],
                            "content": env_response.observation,
                        },
                    ]
                )
            else:
                messages.extend(
                    [
                        next_message,
                        {"role": "user", "content": env_response.observation},
                    ]
                )
            if env_response.done:
                break
        return SolveResult(
            reward=reward,
            info=info,
            messages=messages,
            total_cost=total_cost,
        )


def message_to_action(
    message: Dict[str, Any],
) -> Action:
    if "tool_calls" in message and message["tool_calls"] is not None and len(message["tool_calls"]) > 0 and message["tool_calls"][0]["function"] is not None:
        tool_call = message["tool_calls"][0]
        return Action(
            name=tool_call["function"]["name"],
            kwargs=json.loads(tool_call["function"]["arguments"]),
        )
    else:
        return Action(name=RESPOND_ACTION_NAME, kwargs={"content": message["content"]})

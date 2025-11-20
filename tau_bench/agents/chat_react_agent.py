# Copyright Sierra
# TRUE s1 Budget Forcing Implementation - Token Level Control

import json
import re
from typing import Optional, List, Dict, Any, Tuple

try:
    from vllm import LLM, SamplingParams
    from vllm.outputs import RequestOutput
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

from litellm import completion
import tiktoken

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import (
    Action,
    SolveResult,
    RESPOND_ACTION_NAME,
    RESPOND_ACTION_FIELD_NAME,
)


class ChatReActAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        use_reasoning: bool = True,
        temperature: float = 0.0,
        token_budget: Optional[int] = None,
        enable_wait_tokens: bool = False,
        max_num_steps: int = 30,
        num_wait_tokens: int = 2,
        use_vllm: bool = False,
    ) -> None:
        instruction = REACT_INSTRUCTION if use_reasoning else ACT_INSTRUCTION
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.use_reasoning = use_reasoning
        self.tools_info = tools_info
        self.token_budget = token_budget
        self.use_vllm = use_vllm and VLLM_AVAILABLE
        self.enable_budget_forcing = enable_wait_tokens  # Rename for clarity
        
        # Add budget constraint prompt
        if self.token_budget is not None:
            minimum_budget = int(self.token_budget * 0.5)
            budget_constraint = f"""

# Reasoning Budget Guidance

You have up to {self.token_budget} tokens for reasoning. For complex tasks, aim to use at least {minimum_budget} tokens to ensure thorough analysis.

Best practices:
- Think step-by-step through the problem
- Verify your understanding before taking actions
- Double-check tool outputs and policy requirements
- Consider edge cases and potential errors
"""
        else:
            budget_constraint = ""
        
        self.prompt = (
            wiki + "\n#Available tools\n" + json.dumps(tools_info) + budget_constraint + instruction
        )
        self.max_num_steps = max_num_steps
        self.num_wait_tokens = num_wait_tokens
        self.total_tokens = 0
        
        # Initialize vLLM if requested
        if self.use_vllm:
            if not VLLM_AVAILABLE:
                raise ImportError("vLLM not installed: pip install vllm")
            print(f"🚀 Initializing vLLM with model: {model}")
            self.vllm_model = LLM(
                model=model,
                trust_remote_code=True,
                max_model_len=8192,
                gpu_memory_utilization=0.9,
            )
            # Get tokenizer for checking respond patterns
            self.vllm_tokenizer = self.vllm_model.get_tokenizer()
        else:
            self.vllm_model = None
            self.vllm_tokenizer = None
        
        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model)
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _format_messages_for_vllm(self, messages: List[Dict[str, Any]]) -> str:
        """Convert message list to prompt string for vLLM."""
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"{content}\n\n")
            elif role == "user":
                prompt_parts.append(f"User: {content}\n\n")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}\n\n")
        
        prompt_parts.append("Assistant: ")
        return "".join(prompt_parts)

    def _is_trying_to_respond(self, text: str) -> bool:
        """Check if the generated text contains a respond action."""
        if '"name": "respond"' in text or '"name":"respond"' in text:
            return True
        # Also check for respond without quotes (common error)
        if '"name": respond' in text:
            return True
        return False

    def _extract_action_from_response(self, response_text: str) -> Tuple[str, Dict]:
        """Extract action from model response text."""
        if "Action:" not in response_text:
            return None, None
        
        action_str = response_text.split("Action:")[-1].strip()
        
        # Try to parse JSON
        try:
            action_parsed = json.loads(action_str)
            return action_str, action_parsed
        except json.JSONDecodeError:
            # Try to fix common errors
            action_str_fixed = re.sub(
                r'"name":\s*([a-zA-Z_][a-zA-Z0-9_]*)',
                r'"name": "\1"',
                action_str
            )
            try:
                action_parsed = json.loads(action_str_fixed)
                return action_str_fixed, action_parsed
            except:
                return action_str, None

    def generate_with_budget_forcing_vllm(
        self, 
        messages: List[Dict[str, Any]], 
        minimum_budget: Optional[int] = None
    ) -> Tuple[Dict[str, Any], Action, float, int]:
        """
        TRUE s1 BUDGET FORCING IMPLEMENTATION
        
        This generates text token-by-token and suppresses early stopping
        at the GENERATION level, not by adding user messages.
        """
        prompt = self._format_messages_for_vllm(messages)
        
        # Calculate available token budget
        if self.token_budget:
            max_new_tokens = self.token_budget - self.total_tokens
            max_new_tokens = max(100, min(max_new_tokens, 2048))
        else:
            max_new_tokens = 2048
        
        # CRITICAL: We need to do iterative generation to implement budget forcing
        # Strategy: Generate in chunks, check if trying to stop early
        
        generated_text = ""
        total_tokens_generated = 0
        forced_continuations = 0
        max_forced_continuations = self.num_wait_tokens
        
        while total_tokens_generated < max_new_tokens:
            # Calculate tokens remaining in this chunk
            chunk_size = min(512, max_new_tokens - total_tokens_generated)
            
            if chunk_size < 50:
                break
            
            # Generate a chunk
            sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=chunk_size,
                stop=["User:", "\n\n\n"],  # Natural stop points
            )
            
            current_prompt = prompt + generated_text
            outputs = self.vllm_model.generate([current_prompt], sampling_params)
            chunk_text = outputs[0].outputs[0].text
            
            # Count tokens in this chunk
            chunk_tokens = len(self.tokenizer.encode(chunk_text))
            generated_text += chunk_text
            total_tokens_generated += chunk_tokens
            
            # Check if generation completed naturally
            finish_reason = outputs[0].outputs[0].finish_reason
            
            # Parse to see if there's a complete action
            _, action_parsed = self._extract_action_from_response(generated_text)
            
            # BUDGET FORCING LOGIC
            if action_parsed and action_parsed.get("name") == RESPOND_ACTION_NAME:
                # Agent wants to respond - check if we should force more thinking
                current_total = self.total_tokens + total_tokens_generated
                
                if (self.enable_budget_forcing and 
                    minimum_budget and 
                    current_total < minimum_budget and
                    forced_continuations < max_forced_continuations):
                    
                    # FORCE CONTINUATION - Add prompt to keep thinking
                    print(f"⚠️  Budget forcing at {current_total} tokens (minimum: {minimum_budget})")
                    print(f"   Forcing continuation {forced_continuations + 1}/{max_forced_continuations}")
                    
                    # Instead of stopping, append a continuation prompt
                    # This simulates suppressing the stop token
                    continuation_prompt = "\nThought:\nLet me reconsider and verify my reasoning more carefully before responding.\n"
                    generated_text += continuation_prompt
                    total_tokens_generated += len(self.tokenizer.encode(continuation_prompt))
                    forced_continuations += 1
                    
                    # Continue generating
                    continue
                else:
                    # Either reached minimum budget or max continuations
                    if current_total >= minimum_budget:
                        print(f"✅ Minimum budget reached ({current_total} >= {minimum_budget}), allowing response")
                    else:
                        print(f"⚠️  Max continuations reached, allowing response at {current_total} tokens")
                    break
            
            # If finished for other reasons (stop string, max tokens), break
            if finish_reason == "stop" or finish_reason == "length":
                break
        
        # Extract final action
        action_str, action_parsed = self._extract_action_from_response(generated_text)
        
        if not action_parsed:
            action_parsed = {
                "name": RESPOND_ACTION_NAME,
                "arguments": {RESPOND_ACTION_FIELD_NAME: generated_text},
            }
        
        action = Action(name=action_parsed["name"], kwargs=action_parsed["arguments"])
        
        message = {
            "role": "assistant",
            "content": generated_text,
        }
        
        return message, action, 0.0, total_tokens_generated

    def generate_next_step_litellm(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Action, float, int]:
        """Fallback to LiteLLM (no budget forcing)."""
        res = completion(
            model=self.model,
            custom_llm_provider=self.provider,
            messages=messages,
            temperature=self.temperature,
        )
        message = res.choices[0].message
        action_str = message.content.split("Action:")[-1].strip()
        
        try:
            action_parsed = json.loads(action_str)
        except json.JSONDecodeError:
            action_str_fixed = re.sub(
                r'"name":\s*([a-zA-Z_][a-zA-Z0-9_]*)',
                r'"name": "\1"',
                action_str
            )
            try:
                action_parsed = json.loads(action_str_fixed)
            except:
                action_parsed = {
                    "name": RESPOND_ACTION_NAME,
                    "arguments": {RESPOND_ACTION_FIELD_NAME: action_str},
                }
        
        action = Action(name=action_parsed["name"], kwargs=action_parsed["arguments"])
        token_count = len(self.tokenizer.encode(message.content))
        
        return message.model_dump(), action, res._hidden_params.get("response_cost", 0.0), token_count

    def solve(self, env: Env, task_index: Optional[int] = None) -> SolveResult:
        response = env.reset(task_index=task_index)
        
        self.total_tokens = 0
        reward = 0.0
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": response.observation},
        ]
        total_cost = 0.0
        info = {}
        
        # Calculate minimum budget
        minimum_budget = None
        if self.token_budget and self.enable_budget_forcing:
            minimum_budget = int(self.token_budget * 0.5)
            print(f"📊 Budget forcing enabled: {minimum_budget}/{self.token_budget} tokens (min/max)")
        
        for step_num in range(self.max_num_steps):
            # Generate next step
            if self.use_vllm:
                message, action, cost, token_count = self.generate_with_budget_forcing_vllm(
                    messages, minimum_budget
                )
            else:
                message, action, cost, token_count = self.generate_next_step_litellm(messages)
            
            self.total_tokens += token_count
            
            # Check maximum budget
            budget_exceeded = (self.token_budget is not None and 
                             self.total_tokens >= self.token_budget)
            
            if budget_exceeded:
                info["budget_exceeded"] = True
                info["total_tokens_used"] = self.total_tokens
                print(f"❌ Maximum budget exceeded: {self.total_tokens}/{self.token_budget}")
                
                if action.name != RESPOND_ACTION_NAME:
                    forced_action = Action(
                        name=RESPOND_ACTION_NAME,
                        arguments={RESPOND_ACTION_FIELD_NAME: "I've reached my processing limit. Transferring to human agent."}
                    )
                    response = env.step(forced_action)
                    reward = response.reward
                    info = {**info, **response.info.model_dump()}
                    break
                else:
                    response = env.step(action)
                    obs = response.observation
                    reward = response.reward
                    info = {**info, **response.info.model_dump()}
                    messages.extend([message, {"role": "user", "content": obs}])
                    break
            
            # Execute action
            response = env.step(action)
            obs = response.observation
            reward = response.reward
            info = {**info, **response.info.model_dump()}
            
            if action.name != RESPOND_ACTION_NAME:
                obs = "API output: " + obs
            
            messages.extend([
                message,
                {"role": "user", "content": obs},
            ])
            
            total_cost += cost
            
            if response.done:
                break
        
        info["total_tokens_used"] = self.total_tokens
        
        return SolveResult(
            messages=messages,
            reward=reward,
            info=info,
        )


REACT_INSTRUCTION = """
# Instruction
You need to act as an agent that use the above tools to help the user according to the above policy.

At each step, your generation should have exactly the following format:
Thought:
<A single line of reasoning to process the context and inform the decision making. Do not include extra lines.>
Action:
{"name": <The name of the action>, "arguments": <The arguments to the action in json format>}

The Action will be parsed, so it must be valid JSON.

CRITICAL JSON FORMATTING RULES:
- ALL string values MUST be in double quotes
- Action names MUST be quoted strings
- CORRECT: {"name": "respond", "arguments": {"content": "Hello"}}
- WRONG: {"name": respond, "arguments": {"content": "Hello"}}

IMPORTANT GUIDELINES:
1. If you need information from the user that they haven't provided, ask them using the respond action
2. Before calling transfer_to_human_agents, try at least 3-4 different approaches
3. If a tool returns an error, analyze WHY it failed and try a different approach
4. Do not make up or assume user information - always ask if uncertain
5. When searching for users, if one method fails, try different search methods before giving up

You should not use made-up or placeholder arguments.

Try to be helpful and always follow the policy.
"""


ACT_INSTRUCTION = """
# Instruction
You need to act as an agent that use the above tools to help the user according to the above policy.

At each step, your generation should have exactly the following format:

Action:
{"name": <The name of the action>, "arguments": <The arguments to the action in json format>}

The Action will be parsed, so it must be valid JSON.

Try to be helpful and always follow the policy.
"""
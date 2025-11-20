
import json
import re
from typing import Optional, List, Dict, Any, Tuple

try:
    from vllm import LLM, SamplingParams
    from vllm.outputs import RequestOutput
    VLLM_AVAILABLE = True
except ImportError:
    # If vllm is not available, the agent will fall back to litellm (without budget forcing)
    VLLM_AVAILABLE = False

from litellm import completion
import tiktoken

# Assuming these imports from the tau_bench library
try:
    from tau_bench.agents.base import Agent
    from tau_bench.envs.base import Env
    from tau_bench.types import (
        Action,
        SolveResult,
        RESPOND_ACTION_NAME,
        RESPOND_ACTION_FIELD_NAME,
    )
except ImportError:
    # Placeholder definitions for running the code outside of tau_bench
    class Agent: pass
    class Env: pass
    class Action:
        def __init__(self, name, kwargs):
            self.name = name
            self.kwargs = kwargs
    class SolveResult:
        def __init__(self, messages, reward, info):
            self.messages = messages
            self.reward = reward
            self.info = info
    
    RESPOND_ACTION_NAME = "respond"
    RESPOND_ACTION_FIELD_NAME = "content"


class ChatReActAgent(Agent):
    """
    Implements a ReAct agent with Budget Forcing (BF) mechanisms using vLLM
    for fine-grained token control.

    BF Mechanisms:
    1. Instruction Injection (Budget Guidance) in __init__.
    2. Forced Extension ('Wait' token insertion) in generate_with_budget_forcing_vllm.
    3. Early Termination (Force Answer) in solve method.
    """
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
        gpu_utilization_rate: float = 0.3,
    ) -> None:
        instruction = REACT_INSTRUCTION if use_reasoning else ACT_INSTRUCTION
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.use_reasoning = use_reasoning
        self.tools_info = tools_info
        self.token_budget = token_budget
        self.use_vllm = use_vllm and VLLM_AVAILABLE
        self.enable_budget_forcing = enable_wait_tokens
        self.gpu_utilization_rate = gpu_utilization_rate # Store the new rate
        
        # Mechanism 1: Instruction Injection (Budget Guidance)
        if self.token_budget is not None:
            # KEEPING YOUR 0.1 SETTING FOR DEBUGGING - Revert to 0.5 later for real runs
            minimum_budget = int(self.token_budget * 0.8) 
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
                gpu_memory_utilization=self.gpu_utilization_rate,
            )
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
        """Convert message list to prompt string for vLLM (assuming chat template)."""
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"{content}\n\n")
            elif role == "user":
                prompt_parts.append(f"User: {content}\n\n")
            elif role == "assistant":
                # Only the assistant's thought/action stream is needed for continuation
                prompt_parts.append(f"Assistant: {content}\n\n")
        
        # Critical: Start the new turn for the assistant
        prompt_parts.append("Assistant: ")
        return "".join(prompt_parts)

    def _extract_action_from_response(self, response_text: str) -> Tuple[str, Dict]:
        """Extract action (Thought/Action block) from model response text."""
        # Look for the start of the action block
        if "Action:" not in response_text:
            return None, None
        
        # Isolate the action string
        action_str = response_text.split("Action:")[-1].strip()
        
        try:
            action_parsed = json.loads(action_str)
            return action_str, action_parsed
        except json.JSONDecodeError:
            # Attempt simple fix for unquoted name (e.g., {"name": respond})
            action_str_fixed = re.sub(
                r'"name":\s*([a-zA-Z_][a-zA-Z0-9_]*)',
                r'"name": "\1"',
                action_str
            )
            try:
                action_parsed = json.loads(action_str_fixed)
                return action_str_fixed, action_parsed
            except:
                # Failed to parse, return original text and None
                return action_str, None

    def generate_with_budget_forcing_vllm(
        self, 
        messages: List[Dict[str, Any]], 
        minimum_budget: Optional[int] = None
    ) -> Tuple[Dict[str, Any], Action, float, int]:
        """
        CORRECT s1 BUDGET FORCING IMPLEMENTATION
        
        Mechanism 2 (Forced Extension):
        If the model emits a stop string (e.g., "Action:") BEFORE meeting the minimum_budget,
        we suppress that stop string and append " Wait" to force continued reasoning.
        """
        prompt = self._format_messages_for_vllm(messages)
        
        if self.token_budget:
            max_new_tokens = self.token_budget - self.total_tokens
            max_new_tokens = max(100, min(max_new_tokens, 2048))
        else:
            max_new_tokens = 2048
        
        stop_strings = ["Action:", "\n\n\n"]
        
        generated_text = ""
        total_tokens_generated = 0
        forced_continuations = 0
        max_forced_continuations = self.num_wait_tokens
        
        while total_tokens_generated < max_new_tokens:
            chunk_size = min(2048, max_new_tokens - total_tokens_generated)
            
            if chunk_size < 50:
                break
            
            sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=chunk_size,
                stop=stop_strings,
                skip_special_tokens=False,
                # CRITICAL: Include stop str so we can detect AND strip it manually
                include_stop_str_in_output=True 
            )
            
            current_prompt = prompt + generated_text
            outputs = self.vllm_model.generate([current_prompt], sampling_params)
            chunk_text = outputs[0].outputs[0].text
            finish_reason = outputs[0].outputs[0].finish_reason
            
            chunk_tokens = len(self.tokenizer.encode(chunk_text))
            
            # --- MECHANISM 2: FORCED EXTENSION LOGIC STARTS HERE ---
            # ... (start of suppression logic)
            if finish_reason == "stop":
                current_total = self.total_tokens + total_tokens_generated + chunk_tokens
                
                if (self.enable_budget_forcing and 
                    minimum_budget and 
                    current_total < minimum_budget and
                    forced_continuations < max_forced_continuations):
                    
                    print(f"⚠️  Budget forcing at {current_total} tokens")
                    
                    # FIX 3: Safer Stop String Removal (Only remove from end)
                    cleaned_chunk = chunk_text
                    for stop_str in stop_strings:
                        if cleaned_chunk.endswith(stop_str):
                            cleaned_chunk = cleaned_chunk[: -len(stop_str)].rstrip()
                            break # Only remove one valid stop suffix
                    
                    generated_text += cleaned_chunk
                    total_tokens_generated += len(self.tokenizer.encode(cleaned_chunk))
                    
                    # FIX 1: Use " Wait," (with comma) as per s1 paper
                    wait_token = " Wait," 
                    generated_text += wait_token
                    total_tokens_generated += len(self.tokenizer.encode(wait_token))
                    forced_continuations += 1
                    
                    continue # Loops back to generate() with the new appended "Wait" prompt
                else:
                    # Budget met or max continuations reached -> Allow stop
                    if current_total >= minimum_budget:
                        print(f"✅ Minimum budget reached ({current_total} >= {minimum_budget})")
                    else:
                        print(f"⚠️  Max continuations reached at {current_total} tokens")
                    
                    generated_text += chunk_text
                    total_tokens_generated += chunk_tokens
                    break
            # --- MECHANISM 2 ENDS HERE ---
            
            # Normal generation (length limit or other stop)
            generated_text += chunk_text
            total_tokens_generated += chunk_tokens
            
            if finish_reason == "length":
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
        """Fallback to LiteLLM (no budget forcing implemented)."""
        res = completion(
            model=self.model,
            custom_llm_provider=self.provider,
            messages=messages,
            temperature=self.temperature,
        )
        message = res.choices[0].message
        action_str, action_parsed = self._extract_action_from_response(message.content)
        
        if not action_parsed:
            action_parsed = {
                "name": RESPOND_ACTION_NAME,
                "arguments": {RESPOND_ACTION_FIELD_NAME: message.content},
            }
        
        action = Action(name=action_parsed["name"], kwargs=action_parsed["arguments"])
        token_count = len(self.tokenizer.encode(message.content))
        
        return message.model_dump(), action, res._hidden_params.get("response_cost", 0.0), token_count

    # def solve(self, env: Env, task_index: Optional[int] = None) -> SolveResult:
    #     response = env.reset(task_index=task_index)
        
    #     self.total_tokens = 0
    #     reward = 0.0
    #     messages: List[Dict[str, Any]] = [
    #         {"role": "system", "content": self.prompt},
    #         {"role": "user", "content": response.observation},
    #     ]
    #     total_cost = 0.0
    #     info = {}
        
    #     # Calculate minimum budget for Mechanism 2
    #     minimum_budget = None
    #     if self.token_budget and self.enable_budget_forcing:
    #         minimum_budget = int(self.token_budget * 0.8) 
    #         print(f"📊 Budget forcing enabled: {minimum_budget}/{self.token_budget} tokens (min/max)")
        
    #     for step_num in range(self.max_num_steps):
    #         # Generate next step
    #         if self.use_vllm:
    #             message, action, cost, token_count = self.generate_with_budget_forcing_vllm(
    #                 messages, minimum_budget
    #             )
    #         else:
    #             message, action, cost, token_count = self.generate_next_step_litellm(messages)
            
    #         self.total_tokens += token_count
            
    #         # Mechanism 1: Early Termination (Force Answer)
    #         budget_exceeded = (self.token_budget is not None and 
    #                 self.total_tokens >= self.token_budget)

    #         if budget_exceeded:
    #             info["budget_exceeded"] = True
    #             print(f"❌ Maximum budget exceeded. Forcing immediate action.")
                
    #             # S1 ALIGNMENT: Force the model to output an action now.
    #             # We assume the model was thinking but we cut it off. 
    #             # We append the Action string to force the JSON generation.
                
    #             final_prompt = self._format_messages_for_vllm(messages)
    #             # Append the generated thought trace so far
    #             final_prompt += message['content'] 
    #             # Force the transition to Action
    #             final_prompt += "\nFinal Answer:" 
                
    #             # Generate ONLY the JSON part
    #             sampling_params = SamplingParams(temperature=0, max_tokens=512, stop=["\n"])
    #             outputs = self.vllm_model.generate([final_prompt], sampling_params)
    #             action_json_str = outputs[0].outputs[0].text
                
    #             # Reconstruct the action string for parsing
    #             full_action_str = "Action:" + action_json_str
    #             _, action_parsed = self._extract_action_from_response(full_action_str)
                
    #             if not action_parsed:
    #                 # Fallback if model fails to generate valid JSON even when forced
    #                 forced_action = Action(
    #                     name=RESPOND_ACTION_NAME,
    #                     kwargs={RESPOND_ACTION_FIELD_NAME: "Error: Reasoning budget exceeded and failed to generate valid action."}
    #                 )
    #             else:
    #                 forced_action = Action(name=action_parsed["name"], kwargs=action_parsed["arguments"])

    #             # Execute the forced action
    #             response = env.step(forced_action)
    #             # ... process response as normal ...
    #             break
            
    #         # Execute action
    #         response = env.step(action)
    #         obs = response.observation
    #         reward = response.reward
    #         info = {**info, **response.info.model_dump()}
            
    #         if action.name != RESPOND_ACTION_NAME:
    #             obs = "API output: " + obs
            
    #         messages.extend([
    #             message,
    #             {"role": "user", "content": obs},
    #         ])
            
    #         total_cost += cost if cost is not None else 0.0
            
    #         if response.done:
    #             break
        
    #     info["total_tokens_used"] = self.total_tokens
        
    #     return SolveResult(
    #         messages=messages,
    #         reward=reward,
    #         info=info,
    #     )
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
        
        # Calculate minimum budget for Mechanism 2
        minimum_budget = None
        if self.token_budget and self.enable_budget_forcing:
            minimum_budget = int(self.token_budget * 0.8) 
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
            
            # Mechanism 1: Early Termination (Force Answer)
            budget_exceeded = (self.token_budget is not None and 
                    self.total_tokens >= self.token_budget)

            if budget_exceeded:
                info["budget_exceeded"] = True
                print(f"❌ Maximum budget exceeded. Forcing immediate action.")
                
                # S1 ALIGNMENT: Force the model to output an action now.
                final_prompt = self._format_messages_for_vllm(messages)
                final_prompt += message['content'] 
                final_prompt += "\nFinal Answer:" 
                
                # ---------------------------------------------------------
                # FIX IS HERE: Removed stop=["\n"]. 
                # We stop only at the start of the next user/observation turn.
                # ---------------------------------------------------------
                sampling_params = SamplingParams(
                    temperature=0, 
                    max_tokens=512, 
                    stop=["Observation:", "User:", "API output:"] 
                )
                outputs = self.vllm_model.generate([final_prompt], sampling_params)
                action_json_str = outputs[0].outputs[0].text
                
                # Reconstruct the action string for parsing
                full_action_str = "Action:" + action_json_str
                _, action_parsed = self._extract_action_from_response(full_action_str)
                
                if not action_parsed:
                    forced_action = Action(
                        name=RESPOND_ACTION_NAME,
                        kwargs={RESPOND_ACTION_FIELD_NAME: "Error: Reasoning budget exceeded and failed to generate valid action."}
                    )
                else:
                    forced_action = Action(name=action_parsed["name"], kwargs=action_parsed["arguments"])

                # Execute the forced action
                response = env.step(forced_action)
                reward = response.reward
                info = {**info, **response.info.model_dump()}
                
                # IMPORTANT: Update history so logs are complete
                messages.extend([
                    {"role": "assistant", "content": message['content'] + "\n" + full_action_str},
                    {"role": "user", "content": str(response.observation)}
                ])
                
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
            
            total_cost += cost if cost is not None else 0.0
            
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
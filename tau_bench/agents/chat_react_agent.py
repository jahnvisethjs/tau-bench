# Copyright Sierra

import json
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
from typing import Optional, List, Dict, Any, Tuple


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
    ) -> None:
        instruction = REACT_INSTRUCTION if use_reasoning else ACT_INSTRUCTION
        self.prompt = (
            wiki + "\n#Available tools\n" + json.dumps(tools_info) + instruction
        )
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.use_reasoning = use_reasoning
        self.tools_info = tools_info
        self.token_budget = token_budget
        self.enable_wait_tokens = enable_wait_tokens
        self.max_num_steps = max_num_steps
        self.num_wait_tokens = num_wait_tokens
        self.total_tokens = 0
        # Initialize tokenizer for qwen models
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model)
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def generate_next_step(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[[Dict[str, Any], Action, float], int]:
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
        except json.JSONDecodeError as e:
            # this is a hack
            # Log parsing error for debugging
            print(f"Warning: JSON parsing failed for action: {action_str[:100]}... Error: {e}")
            action_parsed = {
                "name": RESPOND_ACTION_NAME,
                "arguments": {RESPOND_ACTION_FIELD_NAME: action_str},
            }
        assert "name" in action_parsed
        assert "arguments" in action_parsed
        action = Action(name=action_parsed["name"], kwargs=action_parsed["arguments"])

        # Validate that the action name is available
        available_tool_names = [tool["function"]["name"] for tool in self.tools_info] + [RESPOND_ACTION_NAME]
        if action_parsed["name"] not in available_tool_names:
            print(f"Warning: Agent attempted to call unknown tool '{action_parsed['name']}'.")
        
        # Count tokens in the response
        token_count = len(self.tokenizer.encode(message.content))
        
        return message.model_dump(), action, res._hidden_params["response_cost"], token_count

    def solve(self, env: Env, task_index: Optional[int] = None) -> SolveResult:
        response = env.reset(task_index=task_index)
            # Reset token counter for new task
        self.total_tokens = 0
        reward = 0.0
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": response.observation},
        ]
        total_cost = 0.0
        info = {}
        # Track consecutive failures to prevent premature giving up
        consecutive_failures = 0
        MAX_CONSECUTIVE_FAILURES = 3
        for _ in range(self.max_num_steps):
            message, action, cost, token_count = self.generate_next_step(messages)            
            response = env.step(action)

            # Check if the action resulted in an error
            if "error" in response.observation.lower() or "not found" in response.observation.lower():
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    # Add a hint message to help the agent
                    messages.append({
                        "role": "system",
                        "content": "You've encountered multiple consecutive errors. Consider: 1) asking the user for more information, 2) trying a different tool, or 3) explaining the situation to the user before transferring."
                    })
                consecutive_failures = 0  # Reset after hint
            else:
                consecutive_failures = 0  # Reset on success

            # Track total tokens and enforce budget
            self.total_tokens += token_count
            if self.token_budget is not None and self.total_tokens >= self.token_budget:
                # Budget forcing: terminate early when budget exceeded
                info["budget_exceeded"] = True
                info["total_tokens_used"] = self.total_tokens
                break
    
            # Wait token appending: extend thinking if budget remains
            if self.enable_wait_tokens and action.name == RESPOND_ACTION_NAME:
                # Check if we have budget remaining
                if self.token_budget is None or self.total_tokens < self.token_budget:
                # Append Wait tokens to encourage continued reasoning
                for _ in range(self.num_wait_tokens):
                    messages.append({"role": "user", "content": "Wait"})                    # Continue the loop to generate more reasoning
                    continue
            
            obs = response.observation
            reward = response.reward
            info = {**info, **response.info.model_dump()}
            if action.name != RESPOND_ACTION_NAME:
                obs = "API output: " + obs

            messages.extend(
                [
                    message,
                    {"role": "user", "content": obs},
                ]
            )

            total_cost += cost if cost is not None else 0
            if response.done:
                break
        # Add total tokens to info
        info["total_tokens_used"] = self.total_tokens
        
        return SolveResult(
            messages=messages,
            reward=reward,
            info=info,
        )


REACT_INSTRUCTION = f"""
# Instruction
You need to act as an agent that use the above tools to help the user according to the above policy.

At each step, your generation should have exactly the following format:
Thought:
<A single line of reasoning to process the context and inform the decision making. Do not include extra lines.>
Action:
{{"name": <The name of the action>, "arguments": <The arguments to the action in json format>}}

The Action will be parsed, so it must be valid JSON.

IMPORTANT GUIDELINES:
1. If you need information from the user that they haven't provided, ask them using the respond action
2. Before calling transfer_to_human_agents, try at least 3-4 different approaches
3. If a tool returns an error, analyze WHY it failed and try a different approach
4. Do not make up or assume user information - always ask if uncertain
5. When searching for users, if one method fails, try different search methods before giving up

The Action will be parsed, so it must be valid

You should not use made-up or placeholder arguments.

For example, if the user says "I want to know the current weather of San Francisco", and there is such a tool available
{{
    "type": "function",
    "function": {{
        "name": "get_current_weather",
        "description": "Get the current weather",
        "parameters": {{
            "type": "object",
            "properties": {{
                "location": {{
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                }},
                "format": {{
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use. Infer this from the users location.",
                }},
            }},
            "required": ["location", "format"],
        }},
    }}
}}

Your response can be like this:
Thought:
Since the user asks for the weather of San Francisco in USA, the unit should be in fahrenheit. I can query get_current_weather to get the weather.
Action:
{{"name": "get_current_weather", "arguments": {{"location": "San Francisco, CA", "format": "fahrenheit"}}}}

And if the tool returns "70F", your response can be:
Thought:
I can answer the user now.
Action:
{{"name": {RESPOND_ACTION_NAME}, "arguments": {{"{RESPOND_ACTION_FIELD_NAME}": "The current weather of San Francisco is 70F."}}}}

Try to be helpful and always follow the policy.
"""


ACT_INSTRUCTION = f"""
# Instruction
You need to act as an agent that use the above tools to help the user according to the above policy.

At each step, your generation should have exactly the following format:

Action:
{{"name": <The name of the action>, "arguments": <The arguments to the action in json format>}}

You should not use made-up or placeholder arguments.

The Action will be parsed, so it must be valid JSON.

For example, if the user says "I want to know the current weather of San Francisco", and there is such a tool available
```json
{{
    "type": "function",
    "function": {{
        "name": "get_current_weather",
        "description": "Get the current weather",
        "parameters": {{
            "type": "object",
            "properties": {{
                "location": {{
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                }},
                "format": {{
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use. Infer this from the users location.",
                }},
            }},
            "required": ["location", "format"],
        }},
    }}
}}
```

Your response can be like this:
Action:
{{"name": "get_current_weather", "arguments": {{"location": "San Francisco, CA", "format": "fahrenheit"}}}}

And if the tool returns "70F", your response can be:
Action:
{{"name": {RESPOND_ACTION_NAME}, "arguments": {{"{RESPOND_ACTION_FIELD_NAME}": "The current weather of San Francisco is 70F."}}}}

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.

    
if __name__ == "__main__":
    import argparse
    import os
    from tau_bench.envs import get_env
    from tau_bench.envs.user import UserStrategy
    from tau_bench.types import RunConfig
    from litellm import provider_list
    import datetime
    
    # Budget forcing configuration
    TOKEN_BUDGETS = [250, 500, 1000, 2000]
    WAIT_TOKENS = [0, 1, 2, 3, 4, 5]
    
    parser = argparse.ArgumentParser(description="Run budget forcing grid search")
    parser.add_argument("--config", type=str, required=True, help="Task config name")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--provider", type=str, required=True, choices=provider_list, help="Provider name")
    parser.add_argument("--num-tasks", type=int, default=50, help="Number of tasks to run")
    parser.add_argument("--output-dir", type=str, default="budget_forcing_results", help="Output directory")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"results_{timestamp}.json")
    
    results = []
    total_configs = len(TOKEN_BUDGETS) * len(WAIT_TOKENS)
    current_config = 0
    
    print(f"Starting budget forcing grid search with {total_configs} configurations")
    print(f"Token budgets: {TOKEN_BUDGETS}")
    print(f"Wait tokens: {WAIT_TOKENS}")
    print(f"Tasks per config: {args.num_tasks}")
    print(f"="*80)
    
    for token_budget in TOKEN_BUDGETS:
        for num_wait in WAIT_TOKENS:
            current_config += 1
            print(f"\n[{current_config}/{total_configs}] Running: budget={token_budget}, wait_tokens={num_wait}")
            
            # Create environment
            env = get_env(args.config)
            
            # Create agent with specific configuration
            agent = ChatReActAgent(
                tools_info=env.get_tools_info(),
                model=args.model,
                provider=args.provider,
                token_budget=token_budget,
                num_wait_tokens=num_wait
            )
            
            # Run tasks
            task_results = []
            for task_idx in range(args.num_tasks):
                task = env.reset(user_strategy=UserStrategy.STATIC)
                result = agent.solve(task)
                
                task_result = {
                    "task_id": task_idx,
                    "reward": result.reward,
                    "total_cost": result.total_cost,
                    "num_steps": len(result.trajectory),
                    "budget_exceeded": result.total_cost > token_budget if result.total_cost else False
                }
                task_results.append(task_result)
                
                if (task_idx + 1) % 10 == 0:
                    print(f"  Progress: {task_idx + 1}/{args.num_tasks} tasks completed")
            
            # Calculate statistics
            avg_reward = sum(r["reward"] for r in task_results) / len(task_results)
            avg_cost = sum(r["total_cost"] or 0 for r in task_results) / len(task_results)
            success_rate = sum(1 for r in task_results if r["reward"] > 0) / len(task_results)
            budget_exceeded_rate = sum(1 for r in task_results if r["budget_exceeded"]) / len(task_results)
            
            config_result = {
                "token_budget": token_budget,
                "num_wait_tokens": num_wait,
                "avg_reward": avg_reward,
                "avg_cost": avg_cost,
                "success_rate": success_rate,
                "budget_exceeded_rate": budget_exceeded_rate,
                "task_results": task_results
            }
            results.append(config_result)
            
            print(f"  Avg Reward: {avg_reward:.3f}")
            print(f"  Avg Cost: {avg_cost:.1f} tokens")
            print(f"  Success Rate: {success_rate:.1%}")
            print(f"  Budget Exceeded: {budget_exceeded_rate:.1%}")
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump({
            "config": args.config,
            "model": args.model,
            "provider": args.provider,
            "num_tasks": args.num_tasks,
            "timestamp": timestamp,
            "results": results
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"\nSummary Table:")
    print(f"{'Budget':<10} {'Wait':<6} {'Avg Reward':<12} {'Avg Cost':<12} {'Success%':<10} {'Budget Exceed%':<15}")
    print(f"{'-'*80}")
    for r in results:
        print(f"{r['token_budget']:<10} {r['num_wait_tokens']:<6} {r['avg_reward']:<12.3f} {r['avg_cost']:<12.1f} {r['success_rate']*100:<10.1f} {r['budget_exceeded_rate']*100:<15.1f}")


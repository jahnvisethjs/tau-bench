# Copyright Sierra

import os
import json
import random
import traceback
from math import comb
import multiprocessing
from typing import List, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from tau_bench.envs import get_env
from tau_bench.agents.base import Agent
from tau_bench.types import EnvRunResult, RunConfig
from litellm import provider_list
from tau_bench.envs.user import UserStrategy


def run(config: RunConfig) -> List[EnvRunResult]:
    assert config.env in ["retail", "airline"], "Only retail and airline envs are supported"
    assert config.model_provider in provider_list, "Invalid model provider"
    assert config.user_model_provider in provider_list, "Invalid user model provider"
    assert config.agent_strategy in ["tool-calling", "act", "react", "few-shot", "chat-react"], "Invalid agent strategy"
    assert config.task_split in ["train", "test", "dev"], "Invalid task split"
    assert config.user_strategy in [item.value for item in UserStrategy], "Invalid user strategy"

    random.seed(config.seed)
    time_str = datetime.now().strftime("%m%d%H%M%S")
    ckpt_path = f"{config.log_dir}/{config.agent_strategy}-{config.model.split('/')[-1]}-{config.temperature}_range_{config.start_index}-{config.end_index}_user-{config.user_model}-{config.user_strategy}_{time_str}.json"
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    print(f"Loading user with strategy: {config.user_strategy}")
    env = get_env(
        config.env,
        user_strategy=config.user_strategy,
        user_model=config.user_model,
        user_provider=config.user_model_provider,
        task_split=config.task_split,
    )
    agent = agent_factory(
        tools_info=env.tools_info,
        wiki=env.wiki,
        config=config,
    )
    end_index = (
        len(env.tasks) if config.end_index == -1 else min(config.end_index, len(env.tasks))
    )
    results: List[EnvRunResult] = []
    lock = multiprocessing.Lock()
    if config.task_ids and len(config.task_ids) > 0:
        print(f"Running tasks {config.task_ids} (checkpoint path: {ckpt_path})")
    else:
        print(
            f"Running tasks {config.start_index} to {end_index} (checkpoint path: {ckpt_path})"
    )
    for i in range(config.num_trials):
        if config.task_ids and len(config.task_ids) > 0:
            idxs = config.task_ids
        else:
            idxs = list(range(config.start_index, end_index))
        if config.shuffle:
            random.shuffle(idxs)

        def _run(idx: int) -> EnvRunResult:
            isolated_env = get_env(
                config.env,
                user_strategy=config.user_strategy,
                user_model=config.user_model,
                task_split=config.task_split,
                user_provider=config.user_model_provider,
                task_index=idx,
            )

            print(f"Running task {idx}")
            try:
                res = agent.solve(
                    env=isolated_env,
                    task_index=idx,
                )
                result = EnvRunResult(
                    task_id=idx,
                    reward=res.reward,
                    info=res.info,
                    traj=res.messages,
                    trial=i,
                )
            except Exception as e:
                result = EnvRunResult(
                    task_id=idx,
                    reward=0.0,
                    info={"error": str(e), "traceback": traceback.format_exc()},
                    traj=[],
                    trial=i,
                )
            print(
                "✅" if result.reward == 1 else "❌",
                f"task_id={idx}",
                result.info,
            )
            print("-----")
            with lock:
                data = []
                if os.path.exists(ckpt_path):
                    with open(ckpt_path, "r") as f:
                        data = json.load(f)
                with open(ckpt_path, "w") as f:
                    json.dump(data + [result.model_dump()], f, indent=2)
            return result

        with ThreadPoolExecutor(max_workers=config.max_concurrency) as executor:
            res = list(executor.map(_run, idxs))
            results.extend(res)

    display_metrics(results)

    with open(ckpt_path, "w") as f:
        json.dump([result.model_dump() for result in results], f, indent=2)
        print(f"\n📄 Results saved to {ckpt_path}\n")
    return results


# def agent_factory(
#     tools_info: List[Dict[str, Any]], wiki, config: RunConfig
# ) -> Agent:
#     if config.agent_strategy == "tool-calling":
#         # native tool calling
#         from tau_bench.agents.tool_calling_agent import ToolCallingAgent

#         return ToolCallingAgent(
#             tools_info=tools_info,
#             wiki=wiki,
#             model=config.model,
#             provider=config.model_provider,
#             temperature=config.temperature,
#         )
#     elif config.agent_strategy == "act":
#         # `act` from https://arxiv.org/abs/2210.03629
#         from tau_bench.agents.chat_react_agent import ChatReActAgent

#         return ChatReActAgent(
#             tools_info=tools_info,
#             wiki=wiki,
#             model=config.model,
#             provider=config.model_provider,
#             use_reasoning=False,
#             temperature=config.temperature,
#         )
#     elif config.agent_strategy == "react":
#         # `react` from https://arxiv.org/abs/2210.03629
#         from tau_bench.agents.chat_react_agent import ChatReActAgent

#         return ChatReActAgent(
#             tools_info=tools_info,
#             wiki=wiki,
#             model=config.model,
#             provider=config.model_provider,
#             use_reasoning=True,
#             temperature=config.temperature,
#         )
#     elif config.agent_strategy == "few-shot":
#         from tau_bench.agents.few_shot_agent import FewShotToolCallingAgent
#         assert config.few_shot_displays_path is not None, "Few shot displays path is required for few-shot agent strategy"
#         with open(config.few_shot_displays_path, "r") as f:
#             few_shot_displays = [json.loads(line)["messages_display"] for line in f]

#         return FewShotToolCallingAgent(
#             tools_info=tools_info,
#             wiki=wiki,
#             model=config.model,
#             provider=config.model_provider,
#             few_shot_displays=few_shot_displays,
#             temperature=config.temperature,
#         )
#     elif config.agent_strategy == "chat-react":
#                     # ReAct with budget forcing and wait tokens from https://arxiv.org/abs/2501.19393
#         from tau_bench.agents.chat_react_agent import ChatReActAgent
    
#         return ChatReActAgent(
#             tools_info=tools_info,
#             wiki=wiki,
#             model=config.model,
#             provider=config.model_provider,
#             use_reasoning=True,
#             temperature=config.temperature,
#             token_budget=1000,  # Budget forcing with 1000 tokens
#             enable_wait_tokens=True,  # Enable wait token appending
#         )
#     else:
#         raise ValueError(f"Unknown agent strategy: {config.agent_strategy}")


# def display_metrics(results: List[EnvRunResult]) -> None:
#     def is_successful(reward: float) -> bool:
#         return (1 - 1e-6) <= reward <= (1 + 1e-6)

#     num_trials = len(set([r.trial for r in results]))
#     rewards = [r.reward for r in results]
#     avg_reward = sum(rewards) / len(rewards)
#     # c from https://arxiv.org/pdf/2406.12045
#     c_per_task_id: dict[int, int] = {}
#     for result in results:
#         if result.task_id not in c_per_task_id:
#             c_per_task_id[result.task_id] = 1 if is_successful(result.reward) else 0
#         else:
#             c_per_task_id[result.task_id] += 1 if is_successful(result.reward) else 0
#     pass_hat_ks: dict[int, float] = {}
#     for k in range(1, num_trials + 1):
#         sum_task_pass_hat_k = 0
#         for c in c_per_task_id.values():
#             sum_task_pass_hat_k += comb(c, k) / comb(num_trials, k)
#         pass_hat_ks[k] = sum_task_pass_hat_k / len(c_per_task_id)
#     print(f"🏆 Average reward: {avg_reward}")
#     print("📈 Pass^k")
#     for k, pass_hat_k in pass_hat_ks.items():
#         print(f"  k={k}: {pass_hat_k}")
def agent_factory(
    tools_info: List[Dict[str, Any]], wiki, config: RunConfig
) -> Agent:
    if config.agent_strategy == "tool-calling":
        # native tool calling
        from tau_bench.agents.tool_calling_agent import ToolCallingAgent

        return ToolCallingAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "act":
        # `act` from https://arxiv.org/abs/2210.03629
        from tau_bench.agents.chat_react_agent import ChatReActAgent

        return ChatReActAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            use_reasoning=False,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "react":
        # `react` from https://arxiv.org/abs/2210.03629
        from tau_bench.agents.chat_react_agent import ChatReActAgent

        return ChatReActAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            use_reasoning=True,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "few-shot":
        from tau_bench.agents.few_shot_agent import FewShotToolCallingAgent
        assert config.few_shot_displays_path is not None, "Few shot displays path is required for few-shot agent strategy"
        with open(config.few_shot_displays_path, "r") as f:
            few_shot_displays = [json.loads(line)["messages_display"] for line in f]

        return FewShotToolCallingAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            few_shot_displays=few_shot_displays,
            temperature=config.temperature,
        )
    elif config.agent_strategy == "chat-react":
        # ReAct with budget forcing and wait tokens from https://arxiv.org/abs/2501.19393
        from tau_bench.agents.chat_react_agent import ChatReActAgent
        
        # FIX 2: Pass dynamic BF parameters from RunConfig
        return ChatReActAgent(
            tools_info=tools_info,
            wiki=wiki,
            model=config.model,
            provider=config.model_provider,
            use_reasoning=True,
            temperature=config.temperature,
            
            # Use flags parsed from command line
            use_vllm=config.use_vllm,
            token_budget=config.token_budget,
            enable_wait_tokens=config.enable_wait_tokens,
            num_wait_tokens=config.num_wait_tokens,
        )
    else:
        raise ValueError(f"Unknown agent strategy: {config.agent_strategy}")


def display_metrics(results: List[EnvRunResult]) -> None:
    def is_successful(reward: float) -> bool:
        return (1 - 1e-6) <= reward <= (1 + 1e-6)

    num_trials = len(set([r.trial for r in results]))
    rewards = [r.reward for r in results]
    avg_reward = sum(rewards) / len(rewards)
    # c from https://arxiv.org/pdf/2406.12045
    c_per_task_id: dict[int, int] = {}
    for result in results:
        if result.task_id not in c_per_task_id:
            c_per_task_id[result.task_id] = 1 if is_successful(result.reward) else 0
        else:
            c_per_task_id[result.task_id] += 1 if is_successful(result.reward) else 0
    pass_hat_ks: dict[int, float] = {}
    for k in range(1, num_trials + 1):
        sum_task_pass_hat_k = 0
        for c in c_per_task_id.values():
            sum_task_pass_hat_k += comb(c, k) / comb(num_trials, k)
        pass_hat_ks[k] = sum_task_pass_hat_k / len(c_per_task_id)
    print(f"🏆 Average reward: {avg_reward}")
    print("📈 Pass^k")
    for k, pass_hat_k in pass_hat_ks.items():
        print(f"  k={k}: {pass_hat_k}")


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument(
        "--env", type=str, choices=["retail", "airline"], default="retail"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="The model to use for the agent",
    )
    parser.add_argument(
        "--model-provider",
        type=str,
        choices=provider_list,
        help="The model provider for the agent",
    )
    parser.add_argument(
        "--user-model",
        type=str,
        default="gpt-4o",
        help="The model to use for the user simulator",
    )
    parser.add_argument(
        "--user-model-provider",
        type=str,
        choices=provider_list,
        help="The model provider for the user simulator",
    )
    parser.add_argument(
        "--agent-strategy",
        type=str,
        default="tool-calling",
        choices=["tool-calling", "act", "react", "few-shot", "chat-react"],    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The sampling temperature for the action model",
    )
    parser.add_argument(
        "--task-split",
        type=str,
        default="test",
        choices=["train", "test", "dev"],
        help="The split of tasks to run (only applies to the retail domain for now",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=-1, help="Run all tasks if -1")
    parser.add_argument("--task-ids", type=int, nargs="+", help="(Optional) run only the tasks with the given IDs")
    parser.add_argument("--log-dir", type=str, default="results")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Number of tasks to run in parallel",
    )
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--shuffle", type=int, default=0)
    parser.add_argument("--user-strategy", type=str, default="llm", choices=[item.value for item in UserStrategy])
    parser.add_argument("--few-shot-displays-path", type=str, help="Path to a jsonlines file containing few shot displays")
    
    # FIX 1: ADD BUDGET FORCING ARGUMENTS TO ARGPARSE
    parser.add_argument("--use-vllm", action="store_true", help="Enable vLLM integration for token control (required for BF).")
    parser.add_argument("--token-budget", type=int, default=None, help="Maximum tokens allowed for reasoning (Budget Forcing max limit).")
    parser.add_argument("--enable-wait-tokens", action="store_true", help="Enable the 'Wait' token mechanism for forced extension.")
    parser.add_argument("--num-wait-tokens", type=int, default=2, help="Maximum number of 'Wait' tokens to inject.")
    
    args = parser.parse_args()
    print(args)
    return RunConfig(
        model_provider=args.model_provider,
        user_model_provider=args.user_model_provider,
        model=args.model,
        user_model=args.user_model,
        num_trials=args.num_trials,
        env=args.env,
        agent_strategy=args.agent_strategy,
        temperature=args.temperature,
        task_split=args.task_split,
        start_index=args.start_index,
        end_index=args.end_index,
        task_ids=args.task_ids,
        log_dir=args.log_dir,
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        shuffle=args.shuffle,
        user_strategy=args.user_strategy,
        few_shot_displays_path=args.few_shot_displays_path,
        
        # FIX 1: PASS NEW BUDGET FORCING ARGS TO RunConfig
        use_vllm=args.use_vllm,
        token_budget=args.token_budget,
        enable_wait_tokens=args.enable_wait_tokens,
        num_wait_tokens=args.num_wait_tokens,
    )


def main():
    config = parse_args()
    run(config)


if __name__ == "__main__":
    main()
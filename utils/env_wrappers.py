from copy import copy
from collections import defaultdict
from math import cos, sin
from typing import Optional, Tuple

import numpy as np
from gymnasium.spaces import Box, Dict
from metadrive.utils import get_np_random, clip
from ray import rllib
from ray.rllib.env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict
from ray.tune.registry import register_env
from gymnasium.wrappers import EnvCompatibility
from ray.util import PublicAPI

import gymnasium as gym
from typing import Optional

from ray.util.annotations import DeveloperAPI


def get_rllib_compatible_env(env_class, return_class=False):
    env_name = env_class.__name__

    class MA(env_class, MultiAgentEnv):
        _agent_ids = ["agent{}".format(i) for i in range(100)] + ["{}".format(i) for i in range(10000)] + ["sdc"]

        def __init__(self, config, *args, **kwargs):
            env_class.__init__(self, config, *args, **kwargs)
            MultiAgentEnv.__init__(self)

        @property
        def observation_space(self):
            ret = super(MA, self).observation_space
            if not hasattr(ret, "keys"):
                ret.keys = ret.spaces.keys
            return ret

        @property
        def action_space(self):
            ret = super(MA, self).action_space
            if not hasattr(ret, "keys"):
                ret.keys = ret.spaces.keys
            return ret

        def action_space_sample(self, agent_ids: list = None):
            return self.action_space.sample()

        def reset(self,
                  *,
                  seed: Optional[int] = None,
                  options: Optional[dict] = None, ):
            return env_class.reset(self,seed=0)

    class MA_gymnasium(MA):
        def __init__(self, config, *args, **kwargs):
            MA.__init__(self, config, *args, **kwargs)

        def reset(
                self,
                *,
                seed: Optional[int] = None,
                options: Optional[dict] = None,
        ) -> Tuple[MultiAgentDict, MultiAgentDict]:
            return super().reset(seed=seed,options=options)

    MA_gymnasium.__name__ = env_name
    MA_gymnasium.__qualname__ = env_name
    register_env(env_name, lambda config: MA_gymnasium(config))

    if return_class:
        return env_name, MA_gymnasium

    return env_name


def check_old_gym_env(
        env: Optional[gym.Env] = None, *, step_results=None, reset_results=None
):
    # Check `reset()` results.
    if reset_results is not None:
        if (
                # Result is NOT a tuple?
                not isinstance(reset_results, tuple)
                # Result is a tuple of len!=2?
                or len(reset_results) != 2
                # The second item is a NOT dict (infos)?
                or not isinstance(reset_results[1], dict)
                # Result is a tuple of len=2 and the second item is a dict (infos) and
                # our env does NOT have obs space 2-Tuple with the second space being a
                # dict?
                or (
                env
                and isinstance(env.observation_space, gym.spaces.Tuple)
                and len(env.observation_space.spaces) >= 2
                and isinstance(env.observation_space.spaces[1], gym.spaces.Dict)
        )
        ):
            raise ValueError(
                "The number of values returned from `gym.Env.reset(seed=.., options=..)"
                "` must be 2! Make sure your `reset()` method returns: [obs] and "
                "[infos]."
            )
    # Check `step()` results.
    elif step_results is not None:
        if len(step_results) == 5:
            return False
        else:
            raise ValueError(
                "The number of values returned from `gym.Env.step([action])` must be "
                "5 (new gym.Env API including `truncated` flags)! Make sure your "
                "`step()` method returns: [obs], [reward], [terminated], "
                "[truncated], and [infos]!"
            )

    else:
        raise AttributeError(
            "Either `step_results` or `reset_results` most be provided to "
            "`check_old_gym_env()`!"
        )
    return False
def _check_reward(reward, base_env=False, agent_ids=None):
    if base_env:
        for _, multi_agent_dict in reward.items():
            for agent_id, rew in multi_agent_dict.items():
                if not (
                    np.isreal(rew)
                    and not isinstance(rew, bool)
                    and (
                        np.isscalar(rew)
                        or (isinstance(rew, np.ndarray) and rew.shape == ())
                    )
                ):
                    error = (
                        "Your step function must return rewards that are"
                        f" integer or float. reward: {rew}. Instead it was a "
                        f"{type(rew)}"
                    )
                    raise ValueError(error)
                if not (agent_id in agent_ids or agent_id == "__all__"):
                    error = (
                        f"Your reward dictionary must have agent ids that belong to "
                        f"the environment. Agent_ids recieved from "
                        f"env.get_agent_ids() are: {agent_ids}"
                    )
                    raise ValueError(error)
    elif not (
        np.isreal(reward)
        and not isinstance(reward, bool)
        and (
            np.isscalar(reward)
            or (isinstance(reward, np.ndarray) and reward.shape == ())
        )
    ):
        error = (
            "Your step function must return a reward that is integer or float. "
            "Instead it was a {}".format(type(reward))
        )
        raise ValueError(error)


def _check_done_and_truncated(done, truncated, base_env=False, agent_ids=None):
    for what in ["done", "truncated"]:
        data = done if what == "done" else truncated
        if base_env:
            for _, multi_agent_dict in data.items():
                for agent_id, done_ in multi_agent_dict.items():
                    if not isinstance(done_, (bool, np.bool_)):
                        raise ValueError(
                            f"Your step function must return `{what}s` that are "
                            f"boolean. But instead was a {type(data)}"
                        )
                    if not (agent_id in agent_ids or agent_id == "__all__"):
                        error = (
                            f"Your `{what}s` dictionary must have agent ids that "
                            f"belong to the environment. Agent_ids recieved from "
                            f"env.get_agent_ids() are: {agent_ids}"
                        )
                        raise ValueError(error)
        elif not isinstance(data, (bool, np.bool_)):
            error = (
                f"Your step function must return a `{what}` that is a boolean. But "
                f"instead was a {type(data)}"
            )
            raise ValueError(error)


def _check_info(info, base_env=False, agent_ids=None):
    if base_env:
        for _, multi_agent_dict in info.items():
            for agent_id, inf in multi_agent_dict.items():
                if not isinstance(inf, dict):
                    raise ValueError(
                        "Your step function must return infos that are a dict. "
                        f"instead was a {type(inf)}: element: {inf}"
                    )
                if not (agent_id in agent_ids or agent_id == "__all__"):
                    error = (
                        f"Your dones dictionary must have agent ids that belong to "
                        f"the environment. Agent_ids recieved from "
                        f"env.get_agent_ids() are: {agent_ids}"
                    )
                    raise ValueError(error)
    elif not isinstance(info, dict):
        error = (
            "Your step function must return a info that "
            f"is a dict. element type: {type(info)}. element: {info}"
        )
        raise ValueError(error)
def _check_if_element_multi_agent_dict(env, element, function_string, base_env=False):
    if not isinstance(element, dict):
        if base_env:
            error = (
                f"The element returned by {function_string} contains values "
                f"that are not MultiAgentDicts. Instead, they are of "
                f"type: {type(element)}"
            )
        else:
            error = (
                f"The element returned by {function_string} is not a "
                f"MultiAgentDict. Instead, it is of type: "
                f" {type(element)}"
            )
        raise ValueError(error)
    agent_ids: Set = copy(env.get_agent_ids())
    agent_ids.add("__all__")

    if not all(k in agent_ids for k in element):
        if base_env:
            error = (
                f"The element returned by {function_string} has agent_ids"
                f" that are not the names of the agents in the env."
                f"agent_ids in this\nMultiEnvDict:"
                f" {list(element.keys())}\nAgent_ids in this env:"
                f"{list(env.get_agent_ids())}"
            )
        else:
            error = (
                f"The element returned by {function_string} has agent_ids"
                f" that are not the names of the agents in the env. "
                f"\nAgent_ids in this MultiAgentDict: "
                f"{list(element.keys())}\nAgent_ids in this env:"
                f"{list(env.get_agent_ids())}. You likely need to add the private "
                f"attribute `_agent_ids` to your env, which is a set containing the "
                f"ids of agents supported by your env."
            )
        raise ValueError(error)
def local_test_rllib_check():
    from metadrive.envs.marl_envs import MultiAgentTollgateEnv, MultiAgentIntersectionEnv
    env = get_rllib_compatible_env(MultiAgentTollgateEnv, return_class=True)[1]({"num_agents":2})
    print(isinstance(env, MultiAgentEnv))
    print(hasattr(env, "observation_space"))
    print(hasattr(env, "action_space"))
    print(hasattr(env, "_agent_ids"))
    print(hasattr(env, "_obs_space_in_preferred_format"))
    print(hasattr(env, "_action_space_in_preferred_format"))
    obs_and_infos = env.reset(seed=42, options={})
    print(obs_and_infos)
    check_old_gym_env(reset_results=obs_and_infos)
    reset_obs, reset_infos = obs_and_infos
    sampled_obs = env.observation_space_sample()
    _check_if_element_multi_agent_dict(env, reset_obs, "reset()")
    _check_if_element_multi_agent_dict(
        env, sampled_obs, "env.observation_space_sample()"
    )
    env.observation_space_contains(reset_obs)
    sampled_action = env.action_space_sample(list(reset_obs.keys()))
    _check_if_element_multi_agent_dict(env, sampled_action, "action_space_sample")
    results = env.step(sampled_action)
    check_old_gym_env(step_results=results)
    print("============================")
    print(results)
    next_obs, reward, done, truncated, info = results
    _check_if_element_multi_agent_dict(env, next_obs, "step, next_obs")
    _check_if_element_multi_agent_dict(env, reward, "step, reward")
    _check_if_element_multi_agent_dict(env, done, "step, done")
    _check_if_element_multi_agent_dict(env, truncated, "step, truncated")
    _check_if_element_multi_agent_dict(env, info, "step, info")
    _check_reward(
        {"dummy_env_id": reward}, base_env=True, agent_ids=env.get_agent_ids()
    )
    _check_done_and_truncated(
        {"dummy_env_id": done},
        {"dummy_env_id": truncated},
        base_env=True,
        agent_ids=env.get_agent_ids(),
    )
    _check_info({"dummy_env_id": info}, base_env=True, agent_ids=env.get_agent_ids())


    # env.reset()
    # rllib.utils.check_env(env)
    # env.close()

    # env = get_rllib_compatible_env(MultiAgentIntersectionEnv, return_class=True)[1]({})
    # env.reset()
    # rllib.utils.check_env(env)
    # env.close()


if __name__ == '__main__':
    from metadrive.envs.marl_envs import MultiAgentIntersectionEnv

    local_test_rllib_check()

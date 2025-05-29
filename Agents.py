from collections import defaultdict

import numpy as np

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.numpy import convert_to_numpy, softmax

class AlwaysStationaryRLM(RLModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @override(RLModule)
    def _forward_inference(self, batch, **kwargs):
        ret = [4] * len(batch[Columns.OBS])
        return {Columns.ACTIONS: np.array(ret)}

    @override(RLModule)
    def _forward_exploration(self, batch, **kwargs):
        return self._forward_inference(batch, **kwargs)

    @override(RLModule)
    def _forward_train(self, batch, **kwargs):
        raise NotImplementedError(
            "AlwaysStationaryRLM is not trainable! Make sure you do NOT include it "
            "in your `config.multi_agent(policies_to_train={...})` set."
        )
    @override(RLModule)
    def output_specs_inference(self):
        return [Columns.ACTIONS]

    @override(RLModule)
    def output_specs_exploration(self):
        return [Columns.ACTIONS]


class RandomRLM(RLModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @override(RLModule)
    def _forward_inference(self, batch, **kwargs):
        ret = [self.action_space.sample()] * len(batch[Columns.OBS])
        return {Columns.ACTIONS: np.array(ret)}

    @override(RLModule)
    def _forward_exploration(self, batch, **kwargs):
        return self._forward_inference(batch, **kwargs)

    @override(RLModule)
    def _forward_train(self, batch, **kwargs):
        raise NotImplementedError(
            "AlwaysStationaryRLM is not trainable! Make sure you do NOT include it "
            "in your `config.multi_agent(policies_to_train={...})` set."
        )
    @override(RLModule)
    def output_specs_inference(self):
        return [Columns.ACTIONS]

    @override(RLModule)
    def output_specs_exploration(self):
        return [Columns.ACTIONS]


class HybridRandomRLM(RLModule):
    """
    A hybrid agent that combines trained model actions with random actions based on a threshold.
    If random number < threshold, use model action, else use random action.
    """
    def __init__(self, observation_space=None, action_space=None, model_config=None, **kwargs):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            **kwargs
        )
        model_config = model_config or {}
        self.base_module = model_config.get('base_module')
        self.threshold = model_config.get('threshold', 0.8)
        
    @override(RLModule)
    def _forward_inference(self, batch, **kwargs):
        # Get model's actions
        model_output = self.base_module._forward_inference(batch, **kwargs)
        
        # Handle different types of model outputs
        if Columns.ACTIONS in model_output:
            model_actions = model_output[Columns.ACTIONS]
        elif Columns.ACTION_DIST_INPUTS in model_output:
            logits = convert_to_numpy(model_output[Columns.ACTION_DIST_INPUTS])
            model_actions = np.array([
                np.random.choice(len(logits[0]), p=softmax(logit))
                for logit in logits
            ])
        else:
            raise ValueError("Model output format not recognized")
        
        # Generate random actions
        random_actions = np.array([self.action_space.sample() for _ in range(len(batch[Columns.OBS]))])
        
        # Generate random numbers for decision making
        random_numbers = np.random.random(len(batch[Columns.OBS]))
        
        # Create mask for where to use model actions (True where random_number < threshold)
        use_model_mask = random_numbers < self.threshold
        
        # Combine actions based on mask
        final_actions = np.where(use_model_mask[:, None] if len(model_actions.shape) > 1 else use_model_mask, 
                               model_actions, random_actions)
        
        return {Columns.ACTIONS: final_actions}

    @override(RLModule)
    def _forward_exploration(self, batch, **kwargs):
        return self._forward_inference(batch, **kwargs)

    @override(RLModule)
    def _forward_train(self, batch, **kwargs):
        raise NotImplementedError(
            "HybridRandomRLM is not trainable! Make sure you do NOT include it "
            "in your `config.multi_agent(policies_to_train={...})` set."
        )
        
    @override(RLModule)
    def output_specs_inference(self):
        return [Columns.ACTIONS]

    @override(RLModule)
    def output_specs_exploration(self):
        return [Columns.ACTIONS]


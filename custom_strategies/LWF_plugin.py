from avalanche.training import LearningWithoutForgetting
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin


class LwFPlugin(SupervisedPlugin):
    """Learning without Forgetting plugin.

    LwF uses distillation to regularize the current loss with soft targets
    taken from a previous version of the model.
    When used with multi-headed models, all heads are distilled.
    This version is a modified version of the Avalanche implementation for LwF.
    Here instead of simply adding the scaled distillation loss to the regular cross entropy loss we also
    multiply the cross entropy loss by (1-alpha), which aligns with the procedure from "Online continual learning in image classification: An empirical survey"
    available via: https://arxiv.org/abs/2101.10423
    """

    def __init__(self, alpha=1, temperature=2):
        """
        :param alpha: distillation hyperparameter. It can be either a float
                number or a list containing alpha for each experience.
        :param temperature: softmax temperature for distillation
        """
        super().__init__()
        self.alpha = alpha
        self.experience_id = 0
        self.lwf = LearningWithoutForgetting(alpha, temperature)

    def before_backward(self, strategy, **kwargs):
        """
        Add distillation loss
        """
        alpha = (
            self.alpha[self.experience_id]
            if isinstance(self.alpha, (list, tuple))
            else self.alpha
        )
        strategy.loss = (1-alpha) * strategy.loss + self.lwf(strategy.mb_x, strategy.mb_output, strategy.model)

    def after_training_exp(self, strategy, **kwargs):
        """
        Save a copy of the model after each experience and
        update self.prev_classes to include the newly learned classes.
        """
        self.experience_id += 1
        self.lwf.update(strategy.experience, strategy.model)
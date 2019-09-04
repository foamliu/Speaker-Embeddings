class EmbedderOptimizer(object):
    """A simple wrapper class for learning rate scheduling"""

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.step_num = 0
        self.lr = 0.1

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self._update_lr()
        self.optimizer.step()

    def _update_lr(self):
        self.step_num += 1

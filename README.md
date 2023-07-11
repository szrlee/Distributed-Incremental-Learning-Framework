# Distributed Framework for Incremental and Continual Learning

Suppot the Pascal VOC based partition tasks for the setting of incremental multi label continual learning. (Incremental Multi-Class Multi-Label)

### Our new solution
Modern deep learning techniques are facing the {catastrophic forgetting} problem that
when the deep network is trained on a new task 
without seeing the data in the old task, 
knowledge learned from the old task would be easily forgotten, i.e., test accuracy in old task drops tremendously.

To overcome catastrophic forgetting in a sequence of multi-label classification tasks, it requires multi-label continual learning. We first rethink continual learning problem in the framework of {multi-objective optimization}. Then, we develop the {Projection as Pareto Improvement} (PPI) method within the framework. 
PPI is a strategy for updating the hidden layer
which could locally guarantee better performance on the new task and no-worse performance on the old tasks.
For the case that only task-specific ground truth labels and data are available in the new task, we adopt the model distillation strategy to mimic pseudo soft labels that are related to previous tasks.
We also developed a benchmark for the multi-label continual learning problem based on Pascal VOC from which we hope further research will benefit.
The extensive empirical studies show the supremacy of our approaches.

The techical report can be found in here ([Multi-label Continual Learning - Projection as Pareto Improvement.pdf](https://github.com/szrlee/Distributed-Multi-Label-Continual-Learning/blob/bc33632c9b5404a089c0144838f433e6129e9ccb/Multi-label%20Continual%20Learning%20-%20Projection%20as%20Pareto%20Improvement.pdf)).

### How to run
Require pytorch 0.4.1
To replicate the experiments, execute
```bash
$ source source.sh
$ cd src/
$ ./train_{approach}.sh
```

### Acknowlegement
* Learning without Forgetting ([LwF](https://arxiv.org/abs/1606.09282))
* Memory based Parameter Adaptation ([MbPA](https://arxiv.org/abs/1802.10542))
* Memory Aware Synapses: Learning what (not) to forget ([MAS](https://arxiv.org/abs/1711.09601))
* Gradient Episodic Memory for Continual Learning ([GEM](https://arxiv.org/abs/1706.08840))
* Overcoming catastrophic forgetting with hard attention to the task ([HAT](https://arxiv.org/abs/1801.01423))

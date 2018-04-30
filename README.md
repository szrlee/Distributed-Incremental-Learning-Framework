# Distributed Framework for Incremental and Continual Learning

Currently suppoty *only* the CelebA Dataset for the setting of incremental adding label. (Incremental Multi-Class Multi-Label)

### How to run
To replicate the experiments, execute
```bash
$ source source.sh
$ ./train.sh {lwf, gem, joint_train, fine_tuning}
```

### Acknowlegement
* Learning without Forgetting ([LwF](https://arxiv.org/abs/1606.09282))
* Memory based Parameter Adaptation ([MbPA](https://arxiv.org/abs/1802.10542))
* Memory Aware Synapses: Learning what (not) to forget ([MAS](https://arxiv.org/abs/1711.09601))
* Gradient Episodic Memory for Continual Learning ([GEM](https://arxiv.org/abs/1706.08840))
* Overcoming catastrophic forgetting with hard attention to the task ([HAT](https://arxiv.org/abs/1801.01423))

# Distributed Framework for Incremental and Continual Learning

Currently suppot *only* the Pascal VOC based partition tasks for the setting of incremental multi label continual learning. (Incremental Multi-Class Multi-Label)

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

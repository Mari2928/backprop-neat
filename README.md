# backprop-neat-python

This project implements backpropagation combining NEAT-Python.

### Requirements
This project requires the following packages. Tested with Python 3.11.11.
```text
jax
graphviz
matplotlib
numpy
math
random
neat-python
scikit-learn
```
### Initial setup

First, set the task  in `example.py` to load the dataset. The number of generations can be set in `__main__`.

```python
task = "spiral"
ds = Dataset(task)

if __name__ == '__main__':
    # run with neat
    network, population = init(False, 40)
    network.best_genome = population.run(network.eval_genomes, network.num_generation)
    plot_best_net(network)

    # run with backprop-neat
    network, population = init(True, 40)
    network.best_genome = population.run(network.eval_genomes, network.num_generation)
    plot_best_net(network)
```
### Run experiments
Then, we can simply run experiments on terminal.

```bash
python example.py
```


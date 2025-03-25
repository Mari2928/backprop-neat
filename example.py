import jax
import jax.numpy as jnp
from jax import grad
from dataset import*
from plot import*
import neat
from collections import defaultdict, deque
from sklearn.metrics import accuracy_score
import numpy as np
import math
import copy

# Initial setup
task = "spiral"
ds = Dataset(task)
penalty_connectio_factor = 0.03
learning_rate = 0.01

class NeuralNetwork:
    def __init__(self):
        self.params = None
        self.weights = None
        self.genome = None
        self.best_genome = None
        self.weight_to_conn = None
        self.num_generation = 1
        self.config = None
        self.backprop = False
        self.fitness = 0.0
        self.x_input = []
        self.y_target = []
        self.num_datapoints = len(ds.train_data_points)
        for point in ds.train_data_points:
            self.x_input.append([point.x, point.y])
            self.y_target.append([point.label])
        self.x_input = jnp.array(self.x_input)
        self.y_target = jnp.array(self.y_target)

    def node_id_to_act(self, genome):
        activations = {}
        for node_id, node in genome.nodes.items():
            activations[node_id] = node.activation
        return activations

    def get_activation(self, weight_key, activations):
        acts = []
        connection = self.weight_to_conn[weight_key]
        for node_id in activations.keys():
            if node_id in connection:
                acts.append(activations[node_id])
        return acts[-1] if acts else None

    def forward(self, x, params, genome):
        out = x
        activations = self.node_id_to_act(genome)
        for i in range(1, len(params) + 1):
            weight_key = f'W{i}'
            if weight_key in params:
                activation = self.get_activation(weight_key, activations)
                weight = params[weight_key][0][0]
                x = jnp.dot(out, weight)
                if activation == "tanh":
                    out = jnp.tanh(x)
                elif activation == "sigmoid":
                    out = jax.nn.sigmoid(x)
                elif activation == "relu":
                    out = jax.nn.relu(x)
                elif activation == "gauss":
                    out = jnp.exp(-jnp.square(x))
                elif activation == "sin":
                    out = jnp.sin(x)
                elif activation == "abs":
                    out = jnp.abs(x)
                elif activation == "square":
                    out = jnp.square(x)
                else:
                    print("Error on activation functions: ", activation)
        return out

    def neat_ave_loss(self, genome, config):
        ave_loss = 0.0
        #preds = []
        #labels = []
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for point in ds.train_data_points:
            output = net.activate((point.x, point.y))
            output = 1 / (1 + np.exp(-output[0]))
            output = 1.0 if output > 0.5 else 0.0
            loss = (output - point.label) ** 2
            ave_loss += loss
            #preds.append(output)
            #labels.append(point.label)
        # penalize loss for many connections
        ave_loss /= self.num_datapoints
        connection_count = len(genome.connections)
        penalty_factor = math.sqrt(1 + penalty_connectio_factor * connection_count)
        return ave_loss * penalty_factor#, accuracy_score(labels, preds)
    def eval_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            init_loss = self.neat_ave_loss(genome, config)
            best_loss = init_loss
            genome.fitness = -best_loss

            if self.backprop:   # backprop neat
                self.genome = copy.deepcopy(genome)
                genome = self.backward(genome)
                backprop_loss = self.neat_ave_loss(genome, config)
                best_loss = backprop_loss
                if backprop_loss > init_loss:
                    genome = copy.deepcopy(self.genome) # backprop was useless
                    best_loss = init_loss

            genome.fitness = -best_loss

    def compute_param_gradients(self, genome, x, y):
        def loss_fn(params, x, y, genome_):
            preds = self.forward(x, params, genome_)
            return jnp.mean((preds - y) ** 2)
        return grad(loss_fn)(self.params, x, y, genome)


    def extract_weights(self, genome):
        weights = []
        for connection in genome.connections.values():
            if connection.enabled:
                # Extract weight for each connection
                weights.append((connection.key[0], connection.key[1], connection.weight))
        # Sort weights by input node, then by output node
        weights.sort()

        weight_matrices = {}
        for input_node, output_node, weight in weights:
            if (input_node, output_node) not in weight_matrices:
                weight_matrices[(input_node, output_node)] = {}
            weight_matrices[(input_node, output_node)] = weight
        return weight_matrices

    def create_layered_params(self, graph):
        def topological_sort(graph):
            in_degree = defaultdict(int)
            adj_list = defaultdict(list)

            for (u, v) in graph:
                adj_list[u].append(v)
                in_degree[v] += 1

            zero_in_degree = deque([node for node in adj_list if in_degree[node] == 0])

            ordered_nodes = []
            while zero_in_degree:
                node = zero_in_degree.popleft()
                ordered_nodes.append(node)
                for neighbor in adj_list[node]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        zero_in_degree.append(neighbor)

            return ordered_nodes

        sorted_nodes = topological_sort(graph)

        layered_params = {}
        weight_to_conn = {}
        layer_num = 1
        visited_edges = set()

        for node in sorted_nodes:
            for (input_node, output_node), weight in graph.items():
                if input_node == node and (input_node, output_node) not in visited_edges:
                    # Adjust weight initialization to accommodate input dimension
                    connections_count = sum(1 for _ in graph.items() if _[0][0] == input_node)
                    weight_array = jnp.ones((connections_count, 1)) * weight
                    layered_params[f'W{layer_num}'] = weight_array
                    weight_to_conn[f'W{layer_num}'] = (input_node, output_node)
                    layer_num += 1
                    visited_edges.add((input_node, output_node))

        return layered_params, weight_to_conn

    def apply_grads(self, grads):
        for p_key, g_key in zip(self.params.keys(), grads.keys()):
            #grads[g_key] = self.clip_grads(grads[g_key], max_norm=1.0)
            self.params[p_key] -= learning_rate * grads[g_key]

    def update_weights(self, genome):
        new_weights = {}
        for p_key in self.params.keys():
            if self.weight_to_conn[p_key] not in new_weights:
                new_weights[self.weight_to_conn[p_key]] = self.params[p_key]

        for conn_key, connection in genome.connections.items():
            if conn_key in new_weights:
                weight_array = new_weights[conn_key]
                weight = float(weight_array[0][0])
                if isinstance(weight, (int, float)):
                    connection.weight = weight
                else:
                    print(f"Invalid weight format for connection {conn_key}: {weight}")
        return genome

    def loss_fn(self, x, y, genome):
        preds = self.forward(x, self.params, genome)
        return jnp.mean((preds - y) ** 2)

    def binarize_by_midpoint(self, outputs):
        min_val = np.min(outputs)
        max_val = np.max(outputs)
        midpoint = (min_val + max_val) / 2
        binary_outputs = (outputs >= midpoint).astype(int)
        return binary_outputs

    def eval_accuracy(self, net):
        y_true = []
        y_pred = []
        for point in ds.test_data_points:
            y_true.append(point.label)
            output = net.activate((point.x, point.y))[0]
            output = 1 / (1 + np.exp(-output))
            output = 1.0 if output > 0.5 else 0.0
            y_pred.append(output)
        # y_pred = self.binarize_by_midpoint(y_pred)
        print("accuracy: ", accuracy_score(y_true, y_pred))
        return y_pred

    def clip_grads(self, grads, max_norm):
        norm = jnp.linalg.norm(grads)
        factor = jnp.minimum(1.0, max_norm / (norm + 1e-6))
        return grads * factor
    def backward(self, genome):
        self.weights = self.extract_weights(genome)
        layered_params, weight_to_conn = self.create_layered_params(self.weights)
        if not layered_params:
            return genome

        self.params = layered_params
        self.weight_to_conn = weight_to_conn

        for epoch in range(100):
            layered_grads = self.compute_param_gradients(genome, self.x_input, self.y_target)
            self.apply_grads(layered_grads)
            if epoch % 10 == 0:
                current_loss = self.loss_fn(self.x_input, self.y_target, genome)
                #print(f"current_loss: {current_loss}")
        return self.update_weights(genome)

def init(use_backprop, num_generation):
    network = NeuralNetwork()
    network.backprop = use_backprop
    network.num_generation = num_generation
    config_path = 'simple.conf'
    network.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 config_path)
    population = neat.Population(network.config)
    # Add reporters to track progress
    population.add_reporter(neat.StdOutReporter(False))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    return network, population

def plot_best_net(network):
    # visualize the best network
    node_names = create_node_names(network.best_genome, network.config)
    dot = visualize_network_with_activations(network.best_genome, network.config, node_names)
    name = 'backpropneat' if network.backprop else 'neat'
    dot.render(f'results/{task}/{name}/gen{network.num_generation}', view=True)
    #plot classification results
    net = neat.nn.FeedForwardNetwork.create(network.best_genome, network.config)
    predicitons = network.eval_accuracy(net)
    plot(ds.test_data_points, predicitons, f'results/{task}/{name}/gen{network.num_generation}_plot.png', net)

if __name__ == '__main__':
    # run with neat
    network, population = init(False, 40)
    network.best_genome = population.run(network.eval_genomes, network.num_generation)
    plot_best_net(network)

    # run with backprop-neat
    network, population = init(True, 40)
    network.best_genome = population.run(network.eval_genomes, network.num_generation)
    plot_best_net(network)

    # save visualized stats
    # stats = neat.StatisticsReporter()
    # population.add_reporter(stats)
    # visualize.plot_stats(stats, ylog=False, view=True, filename=os.path.join('', "fitness.svg"))
    # visualize.draw_net(network.config, network.best_genome, view=True, filename=os.path.join('', "neat_winner" + "-net.gv"))
    # # plt.plot(sc.episode_score, 'g-', label='score')
    # plt.plot(sc.episode_fitness, 'b-', label='fitness')
    # plt.grid()
    # plt.legend(loc='best')
    # plt.savefig(os.path.join(logdir, "scores.svg"))
    # plt.close()




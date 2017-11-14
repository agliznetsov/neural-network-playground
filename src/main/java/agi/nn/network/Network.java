package agi.nn.network;

import java.util.LinkedList;
import java.util.List;

public class Network {
    private List<List<Node>> nodes = new LinkedList<>();

    private Network() {
    }

    /**
     * Builds a neural nodes.
     *
     * @param networkShape     The shape of the nodes. E.g. [1, 2, 3, 1] means
     *                         the nodes will have one input node, 2 nodes in first hidden layer,
     *                         3 nodes in second hidden layer and 1 output node.
     * @param activation       The activation function of every hidden node.
     * @param outputActivation The activation function for the output nodes.
     * @param regularization   The regularization function that computes a penalty
     *                         for a given weight (parameter) in the nodes. If null, there will be
     *                         no regularization.
     */
    public static Network buildNetwork(
            List<Integer> networkShape,
            ActivationFunction activation,
            ActivationFunction outputActivation,
            RegularizationFunction regularization)

    {
        Network network = new Network();
        int numLayers = networkShape.size();
        int id = 1;
        /** List of layers, with each layer being a list of nodes. */
        for (int layerIdx = 0; layerIdx < numLayers; layerIdx++) {
            boolean isOutputLayer = layerIdx == numLayers - 1;
            boolean isInputLayer = layerIdx == 0;
            List<Node> currentLayer = new LinkedList<>();
            network.nodes.add(currentLayer);
            int numNodes = networkShape.get(layerIdx);
            for (int i = 0; i < numNodes; i++) {
                String nodeId = String.valueOf(id);
                id++;
                Node node = new Node(nodeId, isOutputLayer ? outputActivation : activation);
                currentLayer.add(node);
                if (layerIdx >= 1) {
                    // Add links from nodes in the previous layer to this node.
                    for (int j = 0; j < network.nodes.get(layerIdx - 1).size(); j++) {
                        Node prevNode = network.nodes.get(layerIdx - 1).get(j);
                        Link link = new Link(prevNode, node, regularization);
                        prevNode.outputs.add(link);
                        node.inputLinks.add(link);
                    }
                }
            }
        }
        return network;
    }

    public List<List<Node>> getLayers() {
        return nodes;
    }

    /**
     * Runs a forward propagation of the provided input through the provided
     * nodes. This method modifies the internal state of the nodes - the
     * total input and output of each node in the nodes.
     *
     * @param inputs The input array. Its size() should match the number of input
     *               nodes in the nodes.
     */
    public void forwardProp(double[] inputs) {
        List<Node> inputLayer = nodes.get(0);
        if (inputs == null || inputs.length != inputLayer.size()) {
            throw new Error("The number of inputs must match the number of nodes in the input layer");
        }
        // Update the input layer.
        for (int i = 0; i < inputLayer.size(); i++) {
            Node node = inputLayer.get(i);
            node.output = inputs[i];
        }
        for (int layerIdx = 1; layerIdx < nodes.size(); layerIdx++) {
            List<Node> currentLayer = nodes.get(layerIdx);
            // Update all the nodes in this layer.
            for (int i = 0; i < currentLayer.size(); i++) {
                Node node = currentLayer.get(i);
                node.updateOutput();
            }
        }
    }

    /**
     * Runs a backward propagation using the provided target and the
     * computed output of the previous call to forward propagation.
     * This method modifies the internal state of the nodes - the error
     * derivatives with respect to each node, and each weight
     * in the nodes.
     */
    public void backProp(double[] targets, ErrorFunction errorFunc) {
        // The output node is a special case. We use the user-defined error
        // function for the derivative.
        List<Node> outputLayer = getOutputLayer();
        for(int i=0; i<outputLayer.size(); i++) {
            Node node = outputLayer.get(0);
            node.outputDer = errorFunc.der.applyAsDouble(node.output, targets[i]);
        }

        // Go through the layers backwards.
        for (int layerIdx = nodes.size() - 1; layerIdx >= 1; layerIdx--) {
            List<Node> currentLayer = nodes.get(layerIdx);
            // Compute the error derivative of each node with respect to:
            // 1) its total input
            // 2) each of its input weights.
            for (int i = 0; i < currentLayer.size(); i++) {
                Node node = currentLayer.get(i);
                node.inputDer = node.outputDer * node.activation.der.apply(node.totalInput);
                node.accInputDer += node.inputDer;
                node.numAccumulatedDers++;
            }

            // Error derivative with respect to each weight coming into the node.
            for (int i = 0; i < currentLayer.size(); i++) {
                Node node = currentLayer.get(i);
                for (int j = 0; j < node.inputLinks.size(); j++) {
                    Link link = node.inputLinks.get(j);
                    if (link.isDead) {
                        continue;
                    }
                    link.errorDer = node.inputDer * link.source.output;
                    link.accErrorDer += link.errorDer;
                    link.numAccumulatedDers++;
                }
            }
            if (layerIdx == 1) {
                continue;
            }
            List<Node> prevLayer = nodes.get(layerIdx - 1);
            for (int i = 0; i < prevLayer.size(); i++) {
                Node node = prevLayer.get(i);
                // Compute the error derivative with respect to each node's output.
                node.outputDer = 0;
                for (int j = 0; j < node.outputs.size(); j++) {
                    Link output = node.outputs.get(j);
                    node.outputDer += output.weight * output.dest.inputDer;
                }
            }
        }
    }

    /**
     * Updates the weights of the nodes using the previously accumulated error
     * derivatives.
     */
    public void updateWeights(double learningRate, double regularizationRate) {
        for (int layerIdx = 1; layerIdx < nodes.size(); layerIdx++) {
            List<Node> currentLayer = nodes.get(layerIdx);
            for (int i = 0; i < currentLayer.size(); i++) {
                Node node = currentLayer.get(i);
                // Update the node's bias.
                if (node.numAccumulatedDers > 0) {
                    node.bias -= learningRate * node.accInputDer / node.numAccumulatedDers;
                    node.accInputDer = 0;
                    node.numAccumulatedDers = 0;
                }
                // Update the weights coming into this node.
                for (int j = 0; j < node.inputLinks.size(); j++) {
                    Link link = node.inputLinks.get(j);
                    if (link.isDead) {
                        continue;
                    }
                    double regulDer = link.regularization != RegularizationFunction.None ? link.regularization.der.apply(link.weight) : 0;
                    if (link.numAccumulatedDers > 0) {
                        // Update the weight based on dE/dw.
                        link.weight = link.weight - (learningRate / link.numAccumulatedDers) * link.accErrorDer;
                        // Further update the weight based on regularization.
                        double newLinkWeight = link.weight - (learningRate * regularizationRate) * regulDer;
                        if (link.regularization == RegularizationFunction.L1 && link.weight * newLinkWeight < 0) {
                            // The weight crossed 0 due to the regularization term. Set it to 0.
                            link.weight = 0;
                            link.isDead = true;
                        } else {
                            link.weight = newLinkWeight;
                        }
                        link.accErrorDer = 0;
                        link.numAccumulatedDers = 0;
                    }
                }
            }
        }
    }

    public List<Node> getOutputLayer() {
        return nodes.get(nodes.size() - 1);
    }

}

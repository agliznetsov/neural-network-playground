package agi.nn.network;

import lombok.Data;

import java.util.LinkedList;
import java.util.List;

/**
 * A node in a neural nodes. Each node has a state
 * (total input, output, and their respectively derivatives) which changes
 * after every forward and back propagation run.
 */
public class Node {
    String id;
    List<Link> inputLinks = new LinkedList<>();
    List<Link> outputs = new LinkedList<>();

    double bias = 0.1;
    double totalInput;
    double output;
    /** Error derivative with respect to this node's output. */
    double outputDer = 0;
    /** Error derivative with respect to this node's total input. */
    double inputDer = 0;
    /**
     * Accumulated error derivative with respect to this node's total input since
     * the last update. This derivative equals dE/db where b is the node's
     * bias term.
     */
    double accInputDer = 0;
    /**
     * Number of accumulated err. derivatives with respect to the total input
     * since the last update.
     */
    double numAccumulatedDers = 0;

    /** Activation function that takes total input and returns node's output */
    ActivationFunction activation;

    public double getOutput() {
        return output;
    }

    /**
     * Creates a new node with the provided id and activation function.
     */
    public Node(String id, ActivationFunction activation, boolean initZero) {
        this.id = id;
        this.activation = activation;
        if (initZero) {
            this.bias = 0;
        }
    }

    /** Recomputes the node's output and returns it. */
    public double updateOutput() {
        // Stores total input into the node.
        this.totalInput = this.bias;
        for (int j = 0; j < this.inputLinks.size(); j++) {
            Link link = this.inputLinks.get(j);
            this.totalInput += link.weight * link.source.output;
        }
        this.output = this.activation.output.apply(this.totalInput);
        return this.output;
    }
}
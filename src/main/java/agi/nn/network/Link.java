package agi.nn.network;

import lombok.Data;

/**
 * A link in a neural nodes. Each link has a weight and a source and
 * destination node. Also it has an internal state (error derivative
 * with respect to a particular input) which gets updated after
 * a run of back propagation.
 */
@Data
public class Link {
    String id;
    Node source;
    Node dest;
    double weight = Math.random() - 0.5;
    boolean isDead = false;
    /** Error derivative with respect to this weight. */
    double errorDer = 0;
    /** Accumulated error derivative since the last update. */
    double accErrorDer = 0;
    /** Number of accumulated derivatives since the last update. */
    double numAccumulatedDers = 0;
    RegularizationFunction regularization;

    /**
     * Constructs a link in the neural nodes initialized with random weight.
     *
     * @param source The source node.
     * @param dest The destination node.
     * @param regularization The regularization function that computes the
     *     penalty for this weight. If null, there will be no regularization.
     */
    public Link(Node source, Node dest,
                RegularizationFunction regularization) {
        this.id = source.id + "-" + dest.id;
        this.source = source;
        this.dest = dest;
        this.regularization = regularization;
    }
}
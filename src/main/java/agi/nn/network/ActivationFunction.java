package agi.nn.network;

/**
 * A node's activation function and its derivative.
 */
public enum ActivationFunction {

    LINEAR(x -> x, x -> 1),

    TANH(Math::tanh, x -> {
        double output = Math.tanh(x);
        return 1 - output * output;
    }),

    SIGMOID(x -> 1 / (1 + Math.exp(-x)), x -> {
        double output = 1 / (1 + Math.exp(-x));
        return output * (1 - output);
    }),

    RELU(x -> Math.max(0, x), x -> x <= 0 ? 0 : 1);

    public final DoubleOperator output;
    public final DoubleOperator der;

    ActivationFunction(DoubleOperator output, DoubleOperator der) {
        this.output = output;
        this.der = der;
    }

}

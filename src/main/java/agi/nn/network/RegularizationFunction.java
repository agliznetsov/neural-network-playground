package agi.nn.network;

/**
 * Function that computes a penalty cost for a given weight in the network.
 */
public enum RegularizationFunction {
    None (
            it -> 0,
            it -> 0
    ),

    L1(
            Math::abs,
            w -> w < 0 ? -1 : (w > 0 ? 1 : 0)
    ),

    L2(
            w -> 0.5 * w * w,
            w -> w
    );

    public final DoubleOperator output;
    public final DoubleOperator der;

    RegularizationFunction(DoubleOperator output, DoubleOperator der) {
        this.output = output;
        this.der = der;
    }
}

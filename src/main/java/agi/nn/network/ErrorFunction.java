package agi.nn.network;

import java.util.function.DoubleBinaryOperator;

/**
 * Built-in error functions
 */
public enum ErrorFunction {
    SQUARE(
            (output, target) -> 0.5 * Math.pow(output - target, 2),
            (output, target) -> output - target
    );

    public final DoubleBinaryOperator error;
    public final DoubleBinaryOperator der;

    ErrorFunction(DoubleBinaryOperator error, DoubleBinaryOperator der) {
        this.error = error;
        this.der = der;
    }
}
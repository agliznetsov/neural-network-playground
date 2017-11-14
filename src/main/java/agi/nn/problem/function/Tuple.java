package agi.nn.problem.function;

import agi.nn.problem.Sample;

public class Tuple implements Sample {
    double[] inputs;
    double[] targets;

    public double getX() {
        return inputs[0];
    }

    public double getY() {
        return targets[0];
    }

    @Override
    public double[] getInputs() {
        return inputs;
    }

    @Override
    public double[] getTargets() {
        return targets;
    }

    public Tuple(double x, double y) {
        inputs = new double[]{x};
        targets = new double[]{y};
    }
}

package agi.nn.problem;

import agi.nn.network.ActivationFunction;
import agi.nn.network.Network;
import agi.nn.network.RegularizationFunction;
import agi.nn.problem.function.FunctionProblem;
import agi.nn.problem.points.CircleProblem;
import agi.nn.problem.points.SinusProblem;
import agi.nn.problem.points.SpiralProblem;
import javafx.scene.canvas.Canvas;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public abstract class Problem<T extends Sample> {
    public static final List<Problem> VALUES = Arrays.asList(
            new FunctionProblem("Sinus", x -> Math.sin(x)),
            new FunctionProblem("X*X", x -> Math.pow(x, 2)),
            new CircleProblem(),
            new SpiralProblem()
    );

    public abstract String getName();

    public abstract List<Double> getInputs(T sample);

    public abstract List<T> createSamples(int count);

    public abstract void drawNetwork(Canvas canvas, Network network, List<T> trainData);

    public abstract double getLoss(Network network, T sample);

    public abstract Network buildNetwork(int layers, int nodes, 
                                  ActivationFunction activationFunction, 
                                  RegularizationFunction regularizationFunction);
}

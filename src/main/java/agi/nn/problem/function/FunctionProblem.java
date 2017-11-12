package agi.nn.problem.function;

import agi.nn.network.*;
import agi.nn.problem.Problem;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleFunction;

public class FunctionProblem extends Problem<Tuple> {
    private static final int VISUAL_SIZE = 500;
    private static final int DATA_SIZE = 20;

    private final String name;
    private final DoubleFunction<Double> function;

    public FunctionProblem(String name, DoubleFunction<Double> function) {
        this.name = name;
        this.function = function;
    }

    @Override
    public String toString() {
        return getName();
    }

    @Override
    public String getName() {
        return name;
    }

    @Override
    public Network buildNetwork(int layers, int nodes, ActivationFunction activationFunction, RegularizationFunction regularizationFunction) {
        List<Integer> shape = new ArrayList<>();
        shape.add(1);
        for (int i = 0; i < layers; i++) {
            shape.add(nodes);
        }
        shape.add(1);
        return Network.buildNetwork(shape,
                activationFunction,
                ActivationFunction.TANH,
                regularizationFunction);
    }

    @Override
    public List<Double> getInputs(Tuple sample) {
        return Arrays.asList(sample.x);
    }

    @Override
    public List<Tuple> createSamples(int count) {
        double w = DATA_SIZE * 1.0 / count;
        List<Tuple> samples = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            double x = (i - count / 2) * w;
            double y = function.apply(x);
            samples.add(new Tuple(x, y));
        }
        return samples;
    }

    @Override
    public double getLoss(Network network, Tuple sample) {
        double output = network.getOutputLayer().get(0).getOutput();
        double res = ErrorFunction.SQUARE.error.applyAsDouble(output, sample.getValue());
        return res;
    }

    @Override
    public void drawNetwork(Canvas canvas, Network network, List<Tuple> trainData) {
        canvas.setWidth(VISUAL_SIZE);
        canvas.setHeight(VISUAL_SIZE);
        GraphicsContext gc = canvas.getGraphicsContext2D();
        gc.setStroke(Color.BLACK);
        gc.setFill(Color.WHITE);
        gc.setLineWidth(0.5);
        gc.strokeRect(0, 0, VISUAL_SIZE, VISUAL_SIZE);
        gc.fillRect(0, 0, VISUAL_SIZE, VISUAL_SIZE);

        double w = VISUAL_SIZE / DATA_SIZE;
        List<Node> nodes = network.getOutputLayer();

        gc.setFill(Color.BLACK);
        double[] xPoints = new double[trainData.size()];
        double[] yPoints = new double[trainData.size()];
        for (int i=0;i<trainData.size(); i++) {
            Tuple tuple = trainData.get(i);
            double x = tuple.getX() * w + VISUAL_SIZE / 2;
            double y = VISUAL_SIZE - (tuple.getY() * w + VISUAL_SIZE / 2);
            gc.fillOval(x - 2, +y - 2, 4, 4);
            network.forwardProp(getInputs(tuple));
            double v = nodes.get(0).getOutput();
            xPoints[i] = x;
            yPoints[i] = VISUAL_SIZE - (v * w + VISUAL_SIZE / 2);
        }
        gc.setLineWidth(3);
        gc.setStroke(Color.RED);
        gc.strokePolyline(xPoints, yPoints, xPoints.length);
    }

}

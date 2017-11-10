package agi.nn.playground;

import agi.nn.network.ActivationFunction;
import agi.nn.network.ErrorFunction;
import agi.nn.network.Network;

import java.util.Arrays;
import java.util.List;

public class Playground {
    public static final int NUM_SAMPLES_CLASSIFY = 100;

    public int iter = 0;
    public int batchSize = 10;
    public Point[] trainData;
    public Point[] testData;
    public Network network;
    public double trainLoss = 1;
    public double testLoss = 1;
    public double learningRate = 0.01;
    public double regularizationRate = 0.0;

    public Playground() {
        reset();
    }

    Point[] classifyCircleData(int numSamples) {
        Point[] points = new Point[numSamples];
        int radius = 5;
        int index = 0;

        // Generate positive points inside the circle.
        for (int i = 0; i < numSamples / 2; i++) {
            double r = randUniform(0, radius * 0.5);
            double angle = randUniform(0, 2 * Math.PI);
            double x = r * Math.sin(angle);
            double y = r * Math.cos(angle);
            points[index++] = new Point(x, y, 1);
        }

        // Generate negative points outside the circle.
        for (int i = 0; i < numSamples / 2; i++) {
            double r = randUniform(radius * 0.7, radius);
            double angle = randUniform(0, 2 * Math.PI);
            double x = r * Math.sin(angle);
            double y = r * Math.cos(angle);
            points[index++] = new Point(x, y, -1);
        }

        return points;
    }

    public List<Double> constructInput(double x, double y) {
        return Arrays.asList(x, y);
    }

    double getLoss(Point[] dataPoints) {
        double loss = 0;
        for (int i = 0; i < dataPoints.length; i++) {
            Point dataPoint = dataPoints[i];
            List<Double> input = constructInput(dataPoint.x, dataPoint.y);
            double output = network.forwardProp(input);
            //System.out.println(output + " " + dataPoint.label);
            loss += ErrorFunction.SQUARE.error.applyAsDouble(output, dataPoint.label);
        }
        return loss / dataPoints.length;
    }

    double randUniform(double a, double b) {
        return Math.random() * (b - a) + a;
    }



    public void oneStep() {
        iter++;
        for (int i = 0; i < trainData.length; i++) {
            Point point = trainData[i];
            List<Double> inputs = constructInput(point.x, point.y);
            network.forwardProp(inputs);
            network.backProp(point.label, ErrorFunction.SQUARE);
            if ((i + 1) % batchSize == 0) {
                network.updateWeights(learningRate, regularizationRate);
            }
        }
        // Compute the loss.
        trainLoss = getLoss(trainData);
//        testLoss = getLoss(network, testData);
    }

    public void reset() {
        iter = 0;
        trainLoss = 1;
        trainData = classifyCircleData(NUM_SAMPLES_CLASSIFY);
        network = Network.buildNetwork(Arrays.asList(2, 8, 8, 1), ActivationFunction.RELU, ActivationFunction.TANH, null, false);
    }

    public void batch(int size) {
        for (int i = 0; i < size; i++) {
            oneStep();
//            if (i == 0) {
//                System.out.println("Start loss: " + trainLoss);
//            }
        }
//        System.out.println("End loss: " + trainLoss);
    }

    public void loop() {
        while (true) {
            oneStep();
        }
    }
}

package agi.nn.problem.mnist;

import agi.nn.network.ActivationFunction;
import agi.nn.network.Network;
import agi.nn.network.Node;
import agi.nn.network.RegularizationFunction;
import agi.nn.problem.Problem;
import agi.nn.ui.ChartUtils;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static agi.nn.problem.mnist.MnistUtils.IMAGE_SIZE;

public class MnistProblem extends Problem<Digit> {
    private static final int SAMPLES_COUNT = 100;
    private static final Color[] GRAY_SCALE = ChartUtils.linearGradient(256, Color.WHITE, Color.BLACK);

    @Override
    public String getName() {
        return "MNIST";
    }

    @Override
    public String toString() {
        return getName();
    }

    @Override
    public List<Digit> createSamples() {
        return MnistUtils.loadSamples(new File("c:/mnist"), SAMPLES_COUNT);
    }

    @Override
    public void drawNetwork(Canvas canvas, Network network, List<Digit> trainData) {
        canvas.setWidth((IMAGE_SIZE + 2) * 10);
        canvas.setHeight((IMAGE_SIZE + 2) * 10);
        GraphicsContext gr = canvas.getGraphicsContext2D();
        gr.setFill(Color.WHITE);
        gr.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());
        int i = 0;
        for (int x = 0; x < 10; x++) {
            for (int y = 0; y < 10; y++) {
                drawDigit(gr, x * (IMAGE_SIZE + 2), y * (IMAGE_SIZE + 2), trainData.get(i));
                i++;
            }
        }
    }

    private void drawDigit(GraphicsContext gr, int xs, int ys, Digit digit) {
        for (int x = 0; x < IMAGE_SIZE; x++) {
            for (int y = 0; y < IMAGE_SIZE; y++) {
                double v = digit.getData()[y * IMAGE_SIZE + x];
                gr.setFill(GRAY_SCALE[(int) (v * 255)]);
                gr.fillRect(xs + x, ys + y, 1, 1);
            }
        }
    }

    @Override
    public double getLoss(Network network, Digit sample) {
        //Mean squared error
        List<Node> outputs = network.getOutputLayer();
        double output = 0;
        for (int i = 0; i < outputs.size(); i++) {
            output += Math.pow(outputs.get(0).getOutput() - sample.getTargets()[i], 2);
        }
        return output / outputs.size();
    }

    @Override
    public Network buildNetwork(int layers, int nodes, ActivationFunction activationFunction, RegularizationFunction regularizationFunction) {
        List<Integer> shape = new ArrayList<>();
        shape.add(IMAGE_SIZE * IMAGE_SIZE); //input
//        for (int i = 0; i < layers; i++) {
//            shape.add(nodes);
//        }
        shape.add(IMAGE_SIZE * IMAGE_SIZE); //hidden
        shape.add(10); //output
        return Network.buildNetwork(shape,
                ActivationFunction.RELU,
                ActivationFunction.TANH,
                regularizationFunction);
    }

}

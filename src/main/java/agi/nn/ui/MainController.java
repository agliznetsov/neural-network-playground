package agi.nn.ui;

import agi.nn.network.*;
import agi.nn.problem.Feature;
import agi.nn.problem.Problem;
import agi.nn.problem.Sample;
import agi.nn.problem.points.CircleProblem;
import agi.nn.problem.points.Point;
import agi.nn.problem.points.PointProblem;
import agi.nn.util.Utils;
import javafx.animation.AnimationTimer;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.event.ActionEvent;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.paint.Color;
import javafx.scene.paint.Paint;
import javafx.scene.shape.ArcType;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class MainController {
    private static final int SAMPLES_COUNT = 100;
    private static final long SECOND = 1000000000;
    private static final long FRAME_DURATION = SECOND / 20;
    private static final int NODE_SIZE = 100;
    private static Color[] colorMap = Utils.linearGradient(100, Color.BLUE, Color.WHITE, Color.ORANGE);

    public Button playButton;
    public ComboBox<String> problemBox;
    public ComboBox<Double> regularizationRate;
    public ComboBox<RegularizationFunction> regularization;
    public ComboBox<ActivationFunction> activation;
    public ComboBox<Double> learningRate;
    public ComboBox<Integer> batchSize;
    public ComboBox<Integer> noise;
    public Label iterationLabel;
    public Label lossLabel;
    public Label ipsLabel;
    public Canvas colorMapCanvas;
    public Canvas lossCanvas;
    public Canvas networkCanvas;

    int iteration;
    List<Sample> trainData;
    Network network;
    PointProblem problem;
    List<Feature> features;
    LoopArray trainLossArray = new LoopArray(1000);
    double trainLoss;

    AnimationTimer animationTimer;
    boolean running;
    long start = 0;
    long step = 0;
    double ips = 0;

    public void initialize() {
        System.out.println("init");
        initControls();
        animationTimer = new AnimationTimer() {
            @Override
            public void handle(long now) {
                while (System.nanoTime() - now < FRAME_DURATION) {
                    oneStep();
                    step++;
                }

                long duration = (now - start);
                if (duration > SECOND) {
                    ips = step / (duration * 1.0 / SECOND);
                    step = 0;
                    start = System.nanoTime();
                }

                draw();
            }
        };
        reset();
        draw();
        drawColors();
    }

    private void initControls() {
        ObservableList<Double> ratios = FXCollections.observableArrayList(Arrays.asList(0.00001, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0));

        learningRate.setItems(ratios);
        learningRate.setValue(0.01);

        regularizationRate.setItems(ratios);
        regularizationRate.setValue(0.01);

        ObservableList<Integer> batchItems = FXCollections.observableArrayList(Arrays.asList(1, 10, 20, 30, 40, 50));
        batchSize.setItems(batchItems);
        batchSize.setValue(10);

        ObservableList<Integer> noiseItems = FXCollections.observableArrayList(Arrays.asList(0, 10, 20, 30, 40, 50));
        noise.setItems(noiseItems);
        noise.setValue(0);

        ObservableList<ActivationFunction> activationItems = FXCollections.observableArrayList(ActivationFunction.values());
        activation.setItems(activationItems);
        activation.setValue(ActivationFunction.TANH);

        ObservableList<RegularizationFunction> regItems = FXCollections.observableArrayList(RegularizationFunction.values());
        regularization.setItems(regItems);
        regularization.setValue(RegularizationFunction.None);
    }

    public void onPlay(ActionEvent actionEvent) {
        if (running) {
            stopRun();
        } else {
            startRun();
        }
    }

    public void onReset(ActionEvent actionEvent) {
        if (running) {
            stopRun();
        }
        reset();
        draw();
    }

    public void onStep(ActionEvent actionEvent) {
        if (running) {
            stopRun();
        }
        oneStep();
        draw();
    }

    private void draw() {
        iterationLabel.setText(String.format("%,d", iteration));
        ipsLabel.setText(String.format("%.1f", ips));
        lossLabel.setText(String.format("%.4f", trainLoss));
        ChartUtils.plot(lossCanvas, trainLossArray);
        drawNetwork();
    }

    private void drawNetwork() {
        GraphicsContext gc = networkCanvas.getGraphicsContext2D();
        gc.setFill(Color.WHITE);
        gc.fillRect(0,0, networkCanvas.getWidth(), networkCanvas.getHeight());

        double w = NODE_SIZE / PointProblem.RADIUS / 2;
        for (int x = 0; x < NODE_SIZE; x++) {
            for (int y = 0; y < NODE_SIZE; y++) {
                Point p = new Point((x - NODE_SIZE / 2) / w, (y - NODE_SIZE / 2) / w, 0.0);
                network.forwardProp(Arrays.asList(
                        problem.getFeatures().get(0).getInput(p),
                        problem.getFeatures().get(1).getInput(p)
                ));
                for (int layer = 0; layer < network.getLayers().size(); layer++) {
                    List<Node> nodes = network.getLayers().get(layer);
                    for (int i = 0; i < nodes.size(); i++) {
                        int xs = layer *(NODE_SIZE + 5);
                        int ys = i * (NODE_SIZE + 5);
                        gc.setFill(getColor(nodes.get(i).getOutput()));
                        gc.fillRect(xs + x, ys + y, 1, 1);
                    }
                }
            }
        }

        for (int layer = 0; layer < network.getLayers().size(); layer++) {
            List<Node> nodes = network.getLayers().get(layer);
            for (int i = 0; i < nodes.size(); i++) {
                int xs = layer * (NODE_SIZE + 5);
                int ys = i * (NODE_SIZE + 5);
                gc.setStroke(Color.BLACK);
                gc.strokeRect(xs, ys, NODE_SIZE, NODE_SIZE);

                if (layer == network.getLayers().size() - 1) {
                    for(Sample sample : trainData) {
                        Point point = (Point)sample;
                        double x = point.getX() * w + NODE_SIZE / 2;
                        double y = point.getY() * w + NODE_SIZE / 2;
                        gc.strokeArc(xs + x, ys + y, 2, 2, 0, 360, ArcType.ROUND);
                    }
                }
            }
        }

    }

    private Paint getColor(double value) {
        double index = ActivationFunction.SIGMOID.output.apply(value) * (colorMap.length - 1);
        return colorMap[(int) index];
    }

    private void drawColors() {
        GraphicsContext gc = colorMapCanvas.getGraphicsContext2D();
        double w = colorMapCanvas.getWidth() / colorMap.length;
        for (int i = 0; i < colorMap.length; i++) {
            gc.setStroke(colorMap[i]);
            double x = w * i;
            gc.strokeRect(x, 0, w, colorMapCanvas.getHeight());
        }
        gc.setStroke(Color.BLACK);
        gc.strokeRect(0, 0, colorMapCanvas.getWidth(), colorMapCanvas.getHeight());
    }

    void reset() {
        trainLossArray.clear();
        iteration = 0;
        trainLoss = 1;
        problem = new CircleProblem();
        features = problem.getFeatures();
        trainData = problem.createSamples(SAMPLES_COUNT);
        network = Network.buildNetwork(Arrays.asList(2, 8, 8, 1),
                activation.getValue(),
                ActivationFunction.TANH,
                regularization.getValue(),
                false);
    }

    void stopRun() {
        running = false;
        animationTimer.stop();
        playButton.setText("Run");
    }

    void startRun() {
        running = true;
        start = System.nanoTime();
        animationTimer.start();
        playButton.setText("Stop");
    }

    double getLoss(List<Sample> samples) {
        double loss = 0;
        for (Sample sample : samples) {
            List<Double> inputs = features.stream().map(it -> it.getInput(sample)).collect(Collectors.toList());
            double output = network.forwardProp(inputs);
            loss += ErrorFunction.SQUARE.error.applyAsDouble(output, sample.getValue());
        }
        return loss / samples.size();
    }

    void oneStep() {
        iteration++;
        for (int i = 0; i < trainData.size(); i++) {
            Sample sample = trainData.get(i);
            List<Double> inputs = features.stream().map(it -> it.getInput(sample)).collect(Collectors.toList());
            network.forwardProp(inputs);
            network.backProp(sample.getValue(), ErrorFunction.SQUARE);
            if ((i + 1) % batchSize.getValue() == 0) {
                network.updateWeights(learningRate.getValue(), regularizationRate.getValue());
            }
        }
        trainLoss = getLoss(trainData);
        trainLossArray.add(trainLoss);
//        testLoss = getLoss(network, testData);
    }

}

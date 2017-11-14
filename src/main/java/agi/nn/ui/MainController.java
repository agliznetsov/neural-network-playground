package agi.nn.ui;

import agi.nn.network.ActivationFunction;
import agi.nn.network.ErrorFunction;
import agi.nn.network.Network;
import agi.nn.network.RegularizationFunction;
import agi.nn.problem.Problem;
import agi.nn.problem.Sample;
import javafx.animation.AnimationTimer;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.event.ActionEvent;
import javafx.scene.canvas.Canvas;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;

import java.util.Arrays;
import java.util.List;

public class MainController {
    private static final long SECOND = 1000000000;
    private static final long FRAME_DURATION = SECOND / 20;

    public Button playButton;
    public ComboBox<Problem> problem;
    public ComboBox<Double> regularizationRate;
    public ComboBox<RegularizationFunction> regularization;
    public ComboBox<ActivationFunction> activation;
    public ComboBox<Double> learningRate;
    public ComboBox<Integer> batchSize;
    public ComboBox<Integer> noise;
    public Label iterationLabel;
    public Label lossLabel;
    public Label ipsLabel;
    public Canvas lossCanvas;
    public Canvas networkCanvas;
    public TextField hiddenLayers;
    public TextField nodesCount;
    public Label sampleLabel;
    public Label spsLabel;

    AnimationTimer animationTimer;

    Network network;
    List<Sample> trainData;
    LoopArray trainLossArray = new LoopArray(2000);
    int iteration;
    int sampleIndex;
    int sampleCount;
    double maxLoss;
    double trainLoss;
    boolean running;
    long start = 0;
    long startSampleCount = 0;
    long startIteration = 0;
    double ips = 0;
    double sps = 0;

    public void initialize() {
        System.out.println("init");
        initControls();
        animationTimer = new AnimationTimer() {
            @Override
            public void handle(long now) {
                onTimer(now);
            }
        };
        reset();
    }

    private void initControls() {
        hiddenLayers.setText("1");
        nodesCount.setText("10");

        ObservableList<Problem> problems = FXCollections.observableArrayList(Problem.VALUES);
        problem.setItems(problems);
        problem.setValue(problems.get(0));
        problem.valueProperty().addListener((observable, oldValue, newValue) -> onReset(null));

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

    private void onTimer(long now) {
        while (System.nanoTime() - now < FRAME_DURATION) {
            oneStep();
        }

        long duration = (System.nanoTime() - start);
        if (duration > SECOND) {
            ips = (iteration - startIteration) / (duration * 1.0 / SECOND);
            sps = (startSampleCount - sampleCount) / (duration * 1.0 / SECOND);
            startIteration = 0;
            startSampleCount = 0;
            start = System.nanoTime();
        }

        draw();
    }

    private void draw() {
        iterationLabel.setText(String.format("%,d", iteration));
        ipsLabel.setText(String.format("%.1f", ips));
        sampleLabel.setText(String.format("%06d", sampleIndex));
        spsLabel.setText(String.format("%.1f", sps));
        lossLabel.setText(String.format("%.4f", trainLoss));
        ChartUtils.drawLineChart(lossCanvas, trainLossArray, maxLoss);
        problem.getValue().drawNetwork(networkCanvas, network, trainData);
    }

    void reset() {
        trainLossArray.clear();
        sampleIndex = 0;
        sampleCount = 0;
        iteration = 0;
        startIteration = 0;
        startSampleCount = 0;
        trainLoss = 0;
        maxLoss = 0;
        ips = 0;
        sps = 0;
        int layers = Integer.valueOf(hiddenLayers.getText());
        int nodes = Integer.valueOf(nodesCount.getText());
        trainData = problem.getValue().createSamples();
        network = problem.getValue().buildNetwork(layers, nodes, activation.getValue(), regularization.getValue());
        draw();
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
            network.forwardProp(sample.getInputs());
            loss += problem.getValue().getLoss(network, sample);
        }
        return loss / samples.size();
    }

    void oneStep() {
        Sample sample = trainData.get(sampleIndex);
        network.forwardProp(sample.getInputs());
        network.backProp(sample.getTargets(), ErrorFunction.SQUARE);
        if ((sampleIndex) % batchSize.getValue() == 0) {
            network.updateWeights(learningRate.getValue(), regularizationRate.getValue());
        }
        sampleIndex++;
        if (sampleIndex == trainData.size()) {
            sampleIndex = 0;
            iteration++;

            trainLoss = getLoss(trainData);
            trainLossArray.add(trainLoss);
            maxLoss = Math.max(maxLoss, trainLoss);
        }
    }

}

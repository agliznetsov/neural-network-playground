package agi.nn.ui;

import agi.nn.network.ActivationFunction;
import agi.nn.network.ErrorFunction;
import agi.nn.network.Network;
import agi.nn.network.RegularizationFunction;
import agi.nn.problem.Feature;
import agi.nn.problem.Problem;
import agi.nn.problem.Sample;
import agi.nn.problem.points.CircleProblem;
import javafx.animation.AnimationTimer;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.event.ActionEvent;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class MainController {
    private static final int SAMPLES_COUNT = 100;
    private static final long SECOND = 1000000000;
    private static final long FRAME_DURATION = SECOND / 20;

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

    int iteration;
    List<Sample> trainData;
    Network network;
    Problem problem;
    List<Feature> features;
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
    }

    private void initControls() {
        ObservableList<Double> ratios = FXCollections.observableArrayList(Arrays.asList(0.00001, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0));
        learningRate.setItems(ratios);
        learningRate.setValue(0.01);

        regularizationRate.setItems(ratios);
        regularizationRate.setValue(0.01);

        ObservableList<Integer> batchItems = FXCollections.observableArrayList(Arrays.asList(10, 20, 30, 40, 50));
        batchSize.setItems(batchItems);
        batchSize.setValue(10);
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

    void draw() {
        iterationLabel.setText(String.valueOf(iteration));
        lossLabel.setText(String.valueOf(trainLoss));
        ipsLabel.setText(String.valueOf(ips));
        //TODO: canvas
    }

    void reset() {
        iteration = 0;
        trainLoss = 1;
        problem = new CircleProblem();
        features = problem.getFeatures();
        trainData = problem.createSamples(SAMPLES_COUNT);
        network = Network.buildNetwork(Arrays.asList(2, 8, 8, 1), ActivationFunction.RELU, ActivationFunction.TANH, null, false);
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
//        testLoss = getLoss(network, testData);
    }


    void batch(int size) {
        for (int i = 0; i < size; i++) {
            oneStep();
        }
    }

}

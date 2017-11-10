package agi.nn.ui;

import agi.nn.network.ActivationFunction;
import agi.nn.network.RegularizationFunction;
import javafx.beans.property.DoubleProperty;
import javafx.beans.property.SimpleDoubleProperty;
import javafx.beans.property.SimpleListProperty;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.event.ActionEvent;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;

import java.util.Arrays;
import java.util.List;

public class MainController {
    public Button playButton;
    public ComboBox<String> problem;
    public ComboBox<Double> regularizationRate;
    public ComboBox<RegularizationFunction> regularization;
    public ComboBox<ActivationFunction> activation;
    public ComboBox<Double> learningRate;
    public ComboBox<Integer> batchSize;
    public ComboBox<Integer> noise;

    public void onPlay(ActionEvent actionEvent) {
        playButton.setText("Play".equals(playButton.getText()) ? "Stop" : "Play");
    }

    public void onReset(ActionEvent actionEvent) {
    }

    public void onStep(ActionEvent actionEvent) {
    }

    public void initialize() {
        System.out.println("init");
        ObservableList<Double> values = FXCollections.observableArrayList(Arrays.asList(0.00001, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0));
        learningRate.setItems(values);
        learningRate.setValue(0.01);
    }
}

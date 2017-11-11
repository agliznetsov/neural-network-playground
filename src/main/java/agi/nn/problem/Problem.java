package agi.nn.problem;

import agi.nn.network.Network;
import javafx.scene.canvas.Canvas;

import java.util.List;

public interface Problem<T extends Sample> {
    String getName();

    List<Double> getInputs(T sample);

    List<T> createSamples(int count);

    void drawNetwork(Canvas canvas, Network network, List<T> trainData);
}

package agi.nn.problem.points;

import agi.nn.network.ActivationFunction;
import agi.nn.network.Network;
import agi.nn.network.Node;
import agi.nn.problem.Problem;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;
import javafx.scene.paint.Paint;

import java.util.Arrays;
import java.util.List;

import static agi.nn.ui.ChartUtils.COLOR_MAP;

public abstract class PointProblem implements Problem<Point> {
    private static final int DATA_SIZE = 50;
    private static final int VISUAL_SIZE = 300;

    protected final double radius;

    public PointProblem(double radius) {
        this.radius = radius;
    }

    @Override
    public String toString() {
        return getName();
    }

    @Override
    public List<Double> getInputs(Point sample) {
        return Arrays.asList(sample.x, sample.y);
    }

    @Override
    public void drawNetwork(Canvas canvas, Network network, List<Point> trainData) {
        canvas.setWidth(VISUAL_SIZE);
        canvas.setHeight(VISUAL_SIZE);
        GraphicsContext gc = canvas.getGraphicsContext2D();
        gc.setStroke(Color.BLACK);
        gc.setLineWidth(0.5);
        gc.strokeRect(0, 0, VISUAL_SIZE, VISUAL_SIZE);

        double w = DATA_SIZE / radius / 2;
        double z = VISUAL_SIZE / DATA_SIZE;
        int layer = network.getLayers().size() - 1;
        List<Node> nodes = network.getLayers().get(layer);

        for (int x = 0; x < DATA_SIZE; x++) {
            for (int y = 0; y < DATA_SIZE; y++) {
                Point p = new Point((x - DATA_SIZE / 2) / w, (y - DATA_SIZE / 2) / w, 0.0);
                network.forwardProp(getInputs(p));
                for (int i = 0; i < nodes.size(); i++) {
                    gc.setFill(getColor(nodes.get(i).getOutput()));
                    gc.fillRect(x * z, y * z, z, z);
                }
            }
        }

        for (int i = 0; i < nodes.size(); i++) {
            for(Point point : trainData) {
                double x = point.getX() * w * z + VISUAL_SIZE / 2;
                double y = point.getY() * w * z + VISUAL_SIZE / 2;
                gc.setFill(point.getValue() > 0 ? Color.YELLOW : Color.PURPLE);
                gc.fillOval(x - 2, + y - 2, 4, 4);
                gc.strokeOval(x - 2, + y - 2, 4, 4);
            }
        }
    }

    private Paint getColor(double value) {
        double index = ActivationFunction.SIGMOID.output.apply(value) * (COLOR_MAP.length - 1);
        return COLOR_MAP[(int) index];
    }

}

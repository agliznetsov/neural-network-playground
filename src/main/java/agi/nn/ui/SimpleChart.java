package agi.nn.ui;

import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;

public class SimpleChart {
    GraphicsContext gc;
    double width;
    double height;

    public SimpleChart(GraphicsContext gc, double width, double height) {
        this.gc = gc;
        this.width = width;
        this.height = height;
    }

    public void clear() {
        gc.setFill(Color.WHITE);
        gc.fillRect(-width / 2, -height / 2, width, height);

        gc.setStroke(Color.GRAY);
        gc.setLineWidth(1);
        gc.strokeLine(-width / 2, 0, width / 2, 0);
        gc.strokeLine(0, -height / 2, 0, height / 2);
    }

    public void plot(double xPoints[], double yPoints[], Color color, double max) {
        gc.setStroke(color);
        //TODO: normalize
        xPoints = normalize(xPoints, max, width / 2);
        yPoints = normalize(yPoints, max, height / 2);
        gc.strokePolyline(xPoints, yPoints, xPoints.length);
    }

    private double[] normalize(double[] values, double max, double size) {
        double[] res = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            res[i] = values[i] / max * size;
        }
        return res;
    }
}

package agi.nn.ui;

import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;

public class ChartUtils {

    public static void plot(Canvas canvas, LoopArray data) {
        GraphicsContext gc = canvas.getGraphicsContext2D();
        gc.setFill(Color.WHITE);
        gc.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());
        gc.setStroke(Color.BLACK);
        gc.strokeRect(0, 0, canvas.getWidth(), canvas.getHeight());

        gc.setStroke(Color.BLACK);
        gc.setLineWidth(2);

        double[] xPoints = new double[data.getSize()];
        double[] yPoints = new double[data.getSize()];
        double dx = canvas.getWidth() * 1.0 / data.getSize();
        for (int i = 0; i < xPoints.length; i++) {
            xPoints[i] = dx * i;
            yPoints[i] = (1 - data.get(i)) * canvas.getHeight();
        }
        gc.strokePolyline(xPoints, yPoints, xPoints.length);
    }

}

package agi.nn;

import agi.nn.network.ActivationFunction;
import agi.nn.ui.SimpleChart;
import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Button;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.paint.Color;
import javafx.stage.Stage;

public class FunctionApp extends Application {
    HBox toolBar;
    Canvas canvas;
    GraphicsContext gc;

    @Override
    public void start(Stage stage) {
        initUI(stage);
    }

    private void initUI(Stage stage) {
        BorderPane main = new BorderPane();

        toolBar = new HBox();
        main.setTop(toolBar);

        for (ActivationFunction af : ActivationFunction.values()) {
            addButton(af);
        }

        canvas = new Canvas(500, 500);
        main.setCenter(canvas);
        gc = canvas.getGraphicsContext2D();
        gc.translate(canvas.getWidth() / 2, canvas.getHeight() / 2);
        gc.scale(1, -1);

        Scene scene = new Scene(main);
        stage.setScene(scene);
        stage.show();
    }

    private void plot(ActivationFunction af) {
        SimpleChart chart = new SimpleChart(gc, canvas.getWidth(), canvas.getHeight());
        chart.clear();
        double[] xPoints = new double[500];
        double[] yPoints1 = new double[500];
        double[] yPoints2 = new double[500];
        for (int i = 0; i < 500; i++) {
            double x = i / 500.0 * 4 - 2;
            xPoints[i] = x;
            yPoints1[i] = af.output.apply(xPoints[i]);
            yPoints2[i] = af.der.apply(xPoints[i]);
        }
        chart.plot(xPoints, yPoints1, Color.BLUE, 2);
        chart.plot(xPoints, yPoints2, Color.RED, 2);
    }

    private void addButton(ActivationFunction af) {
        Button btn = new Button();
        btn.setText(af.name());
        btn.setOnAction(e -> plot(af));
        toolBar.getChildren().add(btn);
    }

    public static void main(String[] args) {
        launch(args);
    }
}
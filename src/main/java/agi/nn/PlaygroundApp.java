package agi.nn;

import agi.nn.network.ActivationFunction;
import agi.nn.network.Node;
import agi.nn.playground.Playground;
import agi.nn.util.Utils;
import javafx.animation.AnimationTimer;
import javafx.application.Application;
import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.paint.Color;
import javafx.scene.paint.Paint;
import javafx.stage.Stage;

import java.util.List;

public class PlaygroundApp extends Application {
    private static final long SECOND = 1000000000;
    private static final long FRAME_DURATION = SECOND / 20;

    HBox toolBar;
    Label label;
    Canvas canvas;
    GraphicsContext gc;
    Playground playground = new Playground();
    Color[] colorMap = Utils.linearGradient(100, Color.BLUE, Color.WHITE, Color.ORANGE);
    AnimationTimer animationTimer;
    boolean running;
    long past = 0;

    @Override
    public void start(Stage stage) {
        initUI(stage);
        reset();
    }



    private void initUI(Stage stage) {
        BorderPane main = new BorderPane();

        toolBar = new HBox();
        main.setTop(toolBar);

        addButton("Reset", this::reset);
        addButton("Step", this::step);
        addButton("Start/Stop", this::run);

        label = new Label();
        label.setPadding(new Insets(0, 0, 0, 10));
        toolBar.getChildren().add(label);

        canvas = new Canvas(1200, 800);
        main.setCenter(canvas);
        gc = canvas.getGraphicsContext2D();


        animationTimer = new AnimationTimer() {
            @Override
            public void handle(long now) {
                long step = 0;
                long start = System.nanoTime();
                while (System.nanoTime() - start < FRAME_DURATION) {
                    playground.oneStep();
                    step++;
                }
                draw();

                long duration = (now - past);
                if (duration > SECOND) {
                    System.out.println(duration + " FPS: " + SECOND / duration + " SPF: " + step);
                }
                past = now;
            }
        };

        Scene scene = new Scene(main);
        stage.setScene(scene);
        stage.show();
    }

    private void reset() {
        if (running) {
            stopRun();
        }
        playground.reset();
        draw();
    }

    private void step() {
        playground.oneStep();
        draw();
    }

    private void run() {
        if (running) {
            stopRun();
        } else {
            startRun();
        }
    }

    private void stopRun() {
        running = false;
        animationTimer.stop();
    }

    private void startRun() {
        running = true;
        animationTimer.start();
    }

    private void draw() {
        label.setText("Iteration: " + playground.iter + " Loss: " + playground.trainLoss);
        gc.setFill(Color.WHITE);
        gc.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());
        drawColors();
        drawLoss();
        drawNetwork();
    }

    private void drawLoss() {
        double xs = canvas.getWidth() - 10 - 100;
        gc.setFill(Color.RED);
        gc.fillRect(xs, 40, 100 * (1 - playground.trainLoss), 20);
        gc.setStroke(Color.BLACK);
        gc.strokeRect(xs, 40, 100, 20);
    }

    private void drawColors() {
        double xs = canvas.getWidth() - 10 - 100;
        for(int i = 0; i< colorMap.length; i++) {
            gc.setStroke(colorMap[i]);
            gc.strokeRect(xs + i, 10, 1, 20);
        }
        gc.setStroke(Color.BLACK);
        gc.strokeRect(xs, 10, 100, 20);
    }

    private void drawNetwork() {
        for (int layer = 0; layer < playground.network.getLayers().size(); layer++) {
            List<Node> nodes = playground.network.getLayers().get(layer);
            for (int i = 0; i < nodes.size(); i++) {
                int xs = layer * 55 + 10;
                int ys = i * 55 + 10;
                gc.setStroke(Color.BLACK);
                gc.strokeRect(xs, ys, 50, 50);
            }
        }

        for (int x = 0; x < 50; x++) {
            for (int y = 0; y < 50; y++) {
                playground.network.forwardProp(playground.constructInput((x - 25) / 5.0, (y - 25) / 5.0));
                for (int layer = 0; layer < playground.network.getLayers().size(); layer++) {
                    List<Node> nodes = playground.network.getLayers().get(layer);
                    for (int i = 0; i < nodes.size(); i++) {
                        int xs = layer * 55 + 10;
                        int ys = i * 55 + 10;
                        gc.setFill(getColor(nodes.get(i).getOutput()));
                        gc.fillRect(xs + x, ys + y, 1, 1);
                    }
                }
            }
        }
    }

    private Paint getColor(double value) {
        double index = ActivationFunction.SIGMOID.output.apply(value) * (colorMap.length - 1);
        return colorMap[(int) index];
    }

    private void addButton(String name, Runnable runnable) {
        Button btn = new Button();
        btn.setText(name);
        btn.setOnAction(e -> runnable.run());
        toolBar.getChildren().add(btn);
    }

    public static void main(String[] args) {
        launch(args);
    }
}
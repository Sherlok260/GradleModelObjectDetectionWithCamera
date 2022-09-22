package org.example;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.ImageVisualization;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.util.BufferedImageUtils;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class Main {
    public static void main(String[] args) throws IOException, ModelNotFoundException, MalformedModelException, TranslateException {

        String url = "https://github.com/awslabs/djl/raw/master/examples/src/test/resources/dog_bike_car.jpg";
        String url2 = "https://github.com/awslabs/djl/raw/master/examples/src/test/resources/dog-cat.jpg";
        BufferedImage img = BufferedImageUtils.fromUrl(url);
        BufferedImage img2 = BufferedImageUtils.fromUrl(url2);

        Criteria<BufferedImage, DetectedObjects> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.OBJECT_DETECTION)
                        .setTypes(BufferedImage.class, DetectedObjects.class)
                        .optFilter("backbone", "resnet50")
                        .optProgress(new ProgressBar())
                        .build();

        try (ZooModel<BufferedImage, DetectedObjects> model = ModelZoo.loadModel(criteria)) {
            try (Predictor<BufferedImage, DetectedObjects> predictor = model.newPredictor()) {
                DetectedObjects detection = predictor.predict(img);
                ImageVisualization.drawBoundingBoxes(img, detection);
                ImageIO.write(img, "png", new File("result.png"));
                System.out.println(detection);

                DetectedObjects detection2 = predictor.predict(img2);
                ImageVisualization.drawBoundingBoxes(img2, detection2);
                ImageIO.write(img2, "png", new File("result2.png"));
                System.out.println(detection2);

                model.close();
            }
        }

    }
}
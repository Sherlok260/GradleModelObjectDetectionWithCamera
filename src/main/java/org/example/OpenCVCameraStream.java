package org.example;

import ai.djl.Application;
import ai.djl.Device;
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
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.videoio.VideoCapture;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

@Slf4j
public class OpenCVCameraStream {





    public static Mat frame = null;
    private static HttpStreamServer httpStreamService;
    static VideoCapture videoCapture;
    static Timer tmrVideoProcess;

    public static BufferedImage result(ZooModel model, Mat frame) throws IOException, TranslateException {
        BufferedImage img = httpStreamService.Mat2bufferedImage(frame);
        Predictor<BufferedImage, DetectedObjects> predictor = model.newPredictor();
        DetectedObjects detection = predictor.predict(img);
        ImageVisualization.drawBoundingBoxes(img, detection);

        return img;
    }
    public static void start(ZooModel model) throws IOException{


        videoCapture = new VideoCapture();
        videoCapture.open(0);
        if (!videoCapture.isOpened()) {
            model.close();
            return;
        }

        frame = new Mat();
        httpStreamService = new HttpStreamServer(frame);
        new Thread(httpStreamService).start();

        tmrVideoProcess = new Timer(1, new ActionListener() {
            @SneakyThrows
            public void actionPerformed(ActionEvent e) {
                if (!videoCapture.read(frame)) {
                    tmrVideoProcess.stop();
                }


                BufferedImage img = null;
                try {
                    img = httpStreamService.Mat2bufferedImage(frame);
                    Predictor<BufferedImage, DetectedObjects> predictor = model.newPredictor();
                    DetectedObjects detection = predictor.predict(img);
                    ImageVisualization.drawBoundingBoxes(img, detection);

                    ImageIO.write(img, "png", new File("result3.png"));
                } catch (Exception exp) {
                    exp.printStackTrace();
                }

//                result(model, frame);
                httpStreamService.imag = frame;





            }
        });
        tmrVideoProcess.start();
    }








    public static void main(String[] args) throws IOException, ModelNotFoundException, MalformedModelException, TranslateException {

        String url = "https://github.com/awslabs/djl/raw/master/examples/src/test/resources/dog_bike_car.jpg";
        BufferedImage img = BufferedImageUtils.fromUrl(url);

        Criteria<BufferedImage, DetectedObjects> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.OBJECT_DETECTION)
                        .optDevice(Device.gpu())
                        .setTypes(BufferedImage.class, DetectedObjects.class)
                        .optFilter("backbone", "resnet50")
                        .optProgress(new ProgressBar())
                        .build();

        ZooModel<BufferedImage, DetectedObjects> model = ModelZoo.loadModel(criteria);

//        Predictor<BufferedImage, DetectedObjects> predictor = model.newPredictor();
//        DetectedObjects detection = predictor.predict(img);
//        ImageVisualization.drawBoundingBoxes(img, detection);

//        ImageIO.write(img, "png", new File("result.png"));
//        System.out.println(detection);






        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);//Load opencv native library
        start(model);





    }
}

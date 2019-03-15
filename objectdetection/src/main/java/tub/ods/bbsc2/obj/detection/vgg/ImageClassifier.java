package tub.ods.bbsc2.obj.detection.vgg;

import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import tub.ods.bbdc2.obj.detection.ImageNetLabel;
import tub.ods.bbdc2.obj.detection.Prediction;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.ArrayList;
import java.util.List;

//import nl.ordina.jtech.deeplearning.imagerecognitionwebapp.model.Prediction;
//import org.deeplearning4j.nn.modelimport.
//import org.springframework.stereotype.Component;

class ImageClassifier {
    private static final int HEIGHT = 224;
    private static final int WIDTH = 224;
    private static final int CHANNELS = 3;
    private ComputationGraph vgg16;
    Model model;
    private NativeImageLoader nativeImageLoader;

   public ImageClassifier() {

        try {
            ZooModel zooModel = VGG16.builder().build();
           // model = zooModel.initPretrained(PretrainedType.IMAGENET);



            vgg16 = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);
            //ComputationGraph vgg16 = (ComputationGraph) new VGG16().initPretrained(PretrainedType.IMAGENET);
        } catch (IOException e) {
            e.printStackTrace();
        }
        nativeImageLoader = new NativeImageLoader(HEIGHT, WIDTH, CHANNELS);
    }

    String classify(Frame iamgeFile1) throws Exception {

        Java2DFrameConverter bimConverter = new Java2DFrameConverter();
        BufferedImage image = bimConverter.convert(iamgeFile1);
        ByteArrayOutputStream os = new ByteArrayOutputStream();
        ImageIO.write(image,"jpg",os);
        InputStream is = new ByteArrayInputStream(os.toByteArray());

        INDArray image1 = loadImage(is);
        normalizeImage(image1);


        INDArray output1 = processImage(image1);
        List<Prediction> predictions = decodePredictions(output1);
        return predictionsToString(predictions);
    }


    INDArray[] output = null;

    private INDArray processImage(final INDArray image) {

        int i = 0;
        System.out.println(image.size(i));

        output = vgg16.output(false, image);


        return output[0];
    }

    private INDArray loadImage(final InputStream inputStream) {
        INDArray image = null;
        try {
            image = nativeImageLoader.asMatrix(inputStream);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return image;
    }


    private void normalizeImage(final INDArray image) {
        DataNormalization scaler = new VGG16ImagePreProcessor();
        scaler.transform(image);
    }




    private List<Prediction> decodePredictions(INDArray encodedPredictions) throws Exception {
        List<Prediction> decodedPredictions = new ArrayList<Prediction>();
        int[] top5 = new int[5];
        float[] top5Prob = new float[5];

        ImageNetLabel iml = new ImageNetLabel();
        ArrayList<String> labels = iml.getLabels();
        int i = 0;

        for (INDArray currentBatch = encodedPredictions.getRow(0).dup(); i < 5; ++i) {

            top5[i] = Nd4j.argMax(currentBatch, 1).getInt(0, 0);
            top5Prob[i] = currentBatch.getFloat(0, top5[i]);
            currentBatch.putScalar(0, top5[i], 0.0D);

            System.out.println(top5[i]);
            decodedPredictions.add(new Prediction(labels.get(top5[i]), (top5Prob[i] * 100.0F)));
        }

        return decodedPredictions;
    }

    private String predictionsToString(List<Prediction> predictions) {
        StringBuilder builder = new StringBuilder();
        for (Prediction prediction : predictions) {
            builder.append(prediction.toString());
            builder.append('\n');
        }
        return builder.toString();
    }
}
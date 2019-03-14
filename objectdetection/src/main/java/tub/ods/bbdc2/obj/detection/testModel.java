package tub.ods.bbdc2.obj.detection;

import org.datavec.image.loader.Java2DNativeImageLoader;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.deeplearning4j.zoo.util.imagenet.ImageNetLabels;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

public class testModel {

    public static void runModel(File iamgeFile1, File imageFile2, File modelFile){
        String errorMessage;
        ComputationGraph restoredCNN;



        int tileSize = 224;
        try {
            restoredCNN = KerasModelImport.importKerasModelAndWeights(modelFile.getAbsolutePath(), new int[]{tileSize, tileSize, 3}, false);

        } catch (InvalidKerasConfigurationException ex) {
            errorMessage = "Could not load CNN model: " + ex.getMessage() + "  Cause:  " + ex.getCause();
            System.out.println(errorMessage);
            return;
        } catch (UnsupportedKerasConfigurationException ex) {
            errorMessage = "Could not load CNN model: " + ex.getMessage();
            System.out.println(errorMessage);
            return;
        } catch (IOException ex) {
            errorMessage = "Could not load CNN model: IO Exception";
            System.out.println(errorMessage);
            return;
        }
        if (restoredCNN == null) {
            errorMessage = "CNN model is not valid";
            System.out.println(errorMessage);
            return;
        }
        try {
            System.out.println("*******************************************");
            System.out.println("*******************************************");
            System.out.println("*******************************************");
            System.out.println("Outputting sample predictions for File " + modelFile.getAbsolutePath());

            BufferedImage image = ImageIO.read(iamgeFile1);
            DataNormalization scaler = new ImagePreProcessingScaler(-1.0, 1.0);
            Java2DNativeImageLoader loader = new Java2DNativeImageLoader(tileSize, tileSize, 3);

            INDArray indArray1 = loader.asMatrix(image);
            scaler.transform(indArray1);
            INDArray[] output1 = restoredCNN.output(false, indArray1);


           // Layer layer = restoredCNN.getOutputLayer(0);


            //output1[1].shape();
            //System.out.println(output1[1].shape());
            List<Prediction> predictions = decodePredictions(output1[0]);
            String s = predictionsToString(predictions);
            // output1.getClass();
            System.out.println("Output image 1: " + s);

            INDArray indArray2 = loader.asMatrix(image);
            scaler.transform(indArray2);
            INDArray[] output2 = restoredCNN.output(false, indArray2);
            System.out.println("Output image 2: " + Arrays.toString(output2));




        } catch (IOException ex) {
            errorMessage = "Error Loading File";
            System.out.println(errorMessage);
        }







    }

    private static List<Prediction> decodePredictions(INDArray encodedPredictions) throws IOException {
        List<Prediction> decodedPredictions = new ArrayList<Prediction>();
        int[] top5 = new int[5];
        float[] top5Prob = new float[5];

        AccessibleImageNetLabels al = new AccessibleImageNetLabels();
        List<String> labels =al.labels();
        int i = 0;

        for (INDArray currentBatch = encodedPredictions.getRow(0).dup(); i < 5; ++i) {

            top5[i] = Nd4j.argMax(currentBatch, 1).getInt(0, 0);
            top5Prob[i] = currentBatch.getFloat(0, top5[i]);
            currentBatch.putScalar(0, top5[i], 0.0D);

            decodedPredictions.add(new Prediction(labels.get(top5[i]), (top5Prob[i] * 100.0F)));
        }

        return decodedPredictions;
    }

    public static class AccessibleImageNetLabels extends ImageNetLabels {
        public AccessibleImageNetLabels() throws IOException {
            super();
        }
        public List<String> labels() throws IOException {
            return getLabels();
        }
    }
    private static String predictionsToString(List<Prediction> predictions) {
        StringBuilder builder = new StringBuilder();
        for (Prediction prediction : predictions) {
            builder.append(prediction.toString());
            builder.append('\n');
        }
        return builder.toString();
    }


  /*  private class Example {

        public Example() throws IOException {
        }

        public ArrayList<String> getLabels(){
            if (predictionLabels == null) {
                HashMap<String, ArrayList<String>> jsonMap;
                jsonMap = new ObjectMapper().readValue(this.getClass().getResourceAsStream(jsonResource), HashMap.class);
                predictionLabels = new ArrayList<>(jsonMap.size());
                for (int i = 0; i < jsonMap.size(); i++) {
                    predictionLabels.add(jsonMap.get(String.valueOf(i)).get(1));
                }
            }
            return predictionLabels;
        }
    }*/

}

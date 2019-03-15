package tub.ods.bbdc2.obj.detection;

import org.deeplearning4j.zoo.util.BaseLabels;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;


public class ImageNetLabel extends BaseLabels {


    private ArrayList<String> predictionLabels = null;





    public ImageNetLabel() throws IOException {
        this.predictionLabels = getLabels();
    }

    public ArrayList<String> getLabels() throws IOException {

        if (predictionLabels == null) {


            String file = "/home/hasan/Downloads/synset_words.txt";

            FileInputStream fstream = new FileInputStream(file);
            BufferedReader br = new BufferedReader(new InputStreamReader(fstream));


            String strLine;
            ArrayList<String>  splitted = new ArrayList<String>();

//Read File Line By Line
            while ((strLine = br.readLine()) != null)   {
                // Print the content on the console
               String str[] = strLine.split(" ");
               splitted.add(str[1]);


            }

            predictionLabels = new ArrayList<String>(splitted.size());

//Close the input stream
            fstream.close();


            for (int i = 0; i < splitted.size(); i++) {


                //System.out.println(splitted.get(i));

                predictionLabels.add(splitted.get(i));
            }
        }
        return predictionLabels;
    }

    /**
     * Returns the description of tne nth class in the 1000 classes of ImageNet.
     * @param n
     * @return
     */
    public String getLabel(int n) {
        return predictionLabels.get(n);
    }

}

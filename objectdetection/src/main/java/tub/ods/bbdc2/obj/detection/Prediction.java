package tub.ods.bbdc2.obj.detection;

public class Prediction {

    private String label;
    private double percentage;

    public Prediction(String label, double percentage) {
        this.label = label;
        this.percentage = percentage;
    }

    public void setLabel(final String label) {
        this.label = label;
    }

    public void setPercentage(final double percentage) {
        this.percentage = percentage;
    }

    public String toString() {
        return String.format("%s: %.2f ", this.label, this.percentage);
    }

}
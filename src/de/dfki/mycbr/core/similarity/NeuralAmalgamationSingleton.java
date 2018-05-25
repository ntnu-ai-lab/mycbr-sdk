package de.dfki.mycbr.core.similarity;

import com.google.flatbuffers.FlatBufferBuilder;
import de.dfki.mycbr.core.Project;
import de.dfki.mycbr.core.casebase.Attribute;
import de.dfki.mycbr.core.casebase.Instance;
import de.dfki.mycbr.core.model.AttributeDesc;
import de.dfki.mycbr.core.model.SymbolDesc;
import de.dfki.mycbr.core.similarity.config.MultipleConfig;
import de.dfki.mycbr.util.Pair;
import org.datavec.api.split.StringSplit;
import org.deeplearning4j.nn.modelimport.keras.*;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.SparseFormat;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.exception.Nd4jNoSuchWorkspaceException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.ShapeOffsetResolution;
import org.nd4j.linalg.indexing.conditions.Condition;

import java.io.IOException;
import java.nio.IntBuffer;
import java.util.HashMap;
import java.util.List;
import java.util.Observable;

public class NeuralAmalgamationSingleton {
    private MultiLayerNetwork mln;
    private NeuralAmalgamationSingleton(String modelpath){

        //String configFile = "/home/epic/research/dataGetters/operationalmodel_1.0_cat_cross.json";
        //String weightsFile = "/home/epic/research/dataGetters/operationalmodel_1.0_cat_cross.h5";
        String configFile = modelpath+".json"; //0.88
        String weightsFile = modelpath+".h5";
        mln = null;
        try {
            mln = KerasModelImport.importKerasSequentialModelAndWeights(configFile, weightsFile);
        } catch (InvalidKerasConfigurationException e){
            System.out.println("invalidkerasconf");
        } catch (UnsupportedKerasConfigurationException e){
            System.out.println("unsupportedkerasconf");
        } catch (IOException e ){
            System.out.println("IOexception file path is "+modelpath);
        }

    }
    private static HashMap<String,NeuralAmalgamationSingleton> instances;
    public static NeuralAmalgamationSingleton getInstance(String modelpath){
        if(instances==null) {
            instances = new HashMap<>();
        }
        NeuralAmalgamationSingleton singleton = instances.get(modelpath);
        if(singleton!=null)
            return singleton;

        singleton = new NeuralAmalgamationSingleton(modelpath);
        instances.put(modelpath, singleton);
        return singleton;
    }

    public double getSolution(Instance i){
        INDArray this_inp = getArray(i);
        double this_ret = mln.output(this_inp).getDouble(0);
        return this_ret;
    }

    INDArray getArray(Instance c) {
        double[][] data = new double[1][c.getConcept().getAllAttributeDescs().values().size()];
        int counter = 0;
        for(AttributeDesc attributeDesc : c.getConcept().getAllAttributeDescs().values()){
            Attribute att = c.getAttForDesc(attributeDesc);
            data[0][counter++] = Double.parseDouble(att.getValueAsString());
        }

        /*AttributeDesc windspeedattDesc = c.getConcept().getAllAttributeDescs().get("wind_speed");
        Attribute windspeedAttr = c.getAttForDesc(windspeedattDesc);

        AttributeDesc wind_from_direction1_desc = c.getConcept().getAllAttributeDescs().get("wind_from_direction");
        Attribute wind_from_direction_attr = c.getAttForDesc(wind_from_direction1_desc);

        AttributeDesc wind_effect_desc = c.getConcept().getAllAttributeDescs().get("wind_effect");
        Attribute wind_effect_attr = c.getAttForDesc(wind_effect_desc);

        Double windspeed = Double.parseDouble(windspeedAttr.getValueAsString());
        Double wind_from_direction = Double.parseDouble(wind_from_direction_attr.getValueAsString());
        Double windEffect = Double.parseDouble(wind_effect_attr.getValueAsString());
        double[][] arr = new double[1][3];
        arr[0][0] = windspeed;
        arr[0][1] = wind_from_direction;
        arr[0][2] = windEffect;*/
        INDArray inputarr = Nd4j.create(data);
        return inputarr;
    }
    public Similarity calculateSimilarity(Attribute value1, Attribute value2) throws Exception {
        DataSet ds = new DataSet();
        StringSplit ss = new StringSplit(value1.getValueAsString()+","+value2.getValueAsString());

        double[][] inp = new double[1][3];
        INDArray inputarr = Nd4j.create(inp);

        double[][] inp2 = new double[1][3];
        INDArray inputarr2 = Nd4j.create(inp);

        INDArray output = mln.output(inputarr);
        INDArray output2 = mln.output(inputarr2);
        return Similarity.get(output.getDouble(0)-output2.getDouble(0));
    }
    public INDArray getOutput(INDArray a){
        return mln.output(a);
    }

}

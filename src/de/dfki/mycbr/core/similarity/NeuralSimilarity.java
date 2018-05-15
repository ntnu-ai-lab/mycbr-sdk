package de.dfki.mycbr.core.similarity;

import com.google.flatbuffers.FlatBufferBuilder;
import de.dfki.mycbr.core.Project;
import de.dfki.mycbr.core.casebase.Attribute;
import de.dfki.mycbr.core.model.AttributeDesc;
import de.dfki.mycbr.core.model.SymbolDesc;
import de.dfki.mycbr.core.similarity.config.MultipleConfig;
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
import java.util.List;
import java.util.Observable;

public class NeuralSimilarity extends SimFct {
    private MultiLayerNetwork mln;

    public NeuralSimilarity(Project prj, AttributeDesc desc, String name){
        super(prj,desc,name);
        String configFile = ".json";
        String weightsFile = ".h5";
        mln = null;
        try {
            mln = KerasModelImport.importKerasSequentialModelAndWeights(configFile, weightsFile);
        } catch (InvalidKerasConfigurationException e){

        } catch (UnsupportedKerasConfigurationException e){

        } catch (IOException e ){

        }

    }
    /**
     * Underlying array which holds the similarities of each pair of SymbolAttributes
     */
    protected Similarity[][] sims;

    protected String name;

    /**
     * The description of the given attributes
     */
    protected SymbolDesc desc;

    protected boolean isSymmetric = true;
    protected Project prj;
    protected MultipleConfig mc = MultipleConfig.DEFAULT_CONFIG;

    @Override
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

    @Override
    public boolean isSymmetric() {
        return false;
    }

    @Override
    public String getName() {
        return null;
    }

    @Override
    public void setName(String name) {

    }

    @Override
    public AttributeDesc getDesc() {
        return null;
    }

    @Override
    public void setSymmetric(boolean symmetric) {

    }

    @Override
    public MultipleConfig getMultipleConfig() {
        return null;
    }

    @Override
    public void setMultipleConfig(MultipleConfig mc) {

    }

    @Override
    public Project getProject() {
        return null;
    }

    @Override
    public void clone(AttributeDesc descNEW, boolean active) {

    }

    @Override
    public void update(Observable observable, Object o) {

    }
}

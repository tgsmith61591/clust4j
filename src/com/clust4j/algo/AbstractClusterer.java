package com.clust4j.algo;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Random;
import java.util.UUID;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.kernel.Kernel;
import com.clust4j.log.Log;
import com.clust4j.log.Loggable;
import com.clust4j.utils.Distance;
import com.clust4j.utils.GeometricallySeparable;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.NaNException;
import com.clust4j.utils.Named;
import com.clust4j.utils.SimilarityMetric;
import com.clust4j.utils.VecUtils;

/**
 * 
 * The highest level of cluster abstraction in clust4j, AbstractClusterer
 * provides the interface for classifier clustering (both supervised and unsupervised).
 * It also provides all the functionality for any BaseClustererPlanner classes,
 * data normalizing and logging.
 * 
 * @author Taylor G Smith &lt;tgsmith61591@gmail.com&gt;
 *
 */
public abstract class AbstractClusterer implements Loggable, Named, java.io.Serializable {
	private static final long serialVersionUID = -3623527903903305017L;
	public static boolean DEF_VERBOSE = false;
	public static boolean DEF_SCALE = false;
	
	final static public Random DEF_SEED = new Random();
	final public static GeometricallySeparable DEF_DIST = Distance.EUCLIDEAN;
	final private UUID modelKey;
	
	
	/** Underlying data */
	final protected AbstractRealMatrix data;
	/** Similarity metric */
	private GeometricallySeparable dist;
	/** Seed for any shuffles */
	private final Random seed;
	/** Verbose for heavily logging */
	final private boolean verbose;
	
	
	/** Have any warnings occurred -- volatile because can change */
	volatile private boolean hasWarnings = false;
	

	
	
	/**
	 * Base planner class many clustering algorithms
	 * will extend with static inner classes. Some clustering
	 * algorithms will require more parameters and must provide
	 * the interface for the getting/setting of such parameters.
	 * 
	 * @author Taylor G Smith
	 */
	abstract public static class BaseClustererPlanner {
		abstract public GeometricallySeparable getSep();
		abstract public boolean getScale();
		abstract public Random getSeed();
		abstract public boolean getVerbose();
		abstract public BaseClustererPlanner setScale(final boolean b);
		abstract public BaseClustererPlanner setSeed(final Random rand);
		abstract public BaseClustererPlanner setVerbose(final boolean b);
		abstract public BaseClustererPlanner setSep(final GeometricallySeparable dist);
	}
	
	
	
	
	/**
	 * Base clusterer constructor. Sets up the distance measure,
	 * and if necessary scales data.
	 * @param data
	 * @param planner
	 */
	public AbstractClusterer(AbstractRealMatrix data, BaseClustererPlanner planner) {
		
		this.dist = planner.getSep();
		this.verbose = planner.getVerbose();
		this.modelKey = UUID.randomUUID();
		this.seed = planner.getSeed();
		boolean similarity = this.dist instanceof SimilarityMetric; // Avoid later check
		
		// Handle data, now...
		info("checking input data for NaNs");
		handleData(data);
		
		// Log info
		info("initializing " + getName() + 
				" clustering with " + data.getRowDimension() + 
				" x " + data.getColumnDimension() + " data matrix");
		
		if(this.dist instanceof Kernel) {
			warn("running " + getName() + " in Kernel mode can be an expensive option");
		}
		
		info((similarity ? "similarity" : "distance") + 
				" metric: " + dist.getName());
		
		
		// Scale if needed
		if(!planner.getScale())
			this.data = (AbstractRealMatrix) data.copy();
		else {
			info("normalizing matrix columns (centering and scaling)");
			this.data = scale(data, (AbstractRealMatrix) data.copy());
		}
	} // End constructor
	
	
	
	final private void handleData(final AbstractRealMatrix data) {
		if(data.getRowDimension() == 0)
			throw new IllegalArgumentException("empty data");
		
		if(MatUtils.containsNaN(data)) {
			String error = "NaN in input data. Select a matrix imputation method for incomplete records";
			error(error);
			throw new NaNException(error);
		}
	}
	
	
	
	/**
	 * Static method to scale a matrix
	 * @param data
	 * @return
	 */
	final static protected AbstractRealMatrix scale(AbstractRealMatrix data) {
		return scale(data, (AbstractRealMatrix) data.copy());
	}
	
	
	/**
	 * Static method to scale a matrix given a copy
	 * @param data
	 * @param copy
	 * @return
	 */
	final static protected AbstractRealMatrix scale(AbstractRealMatrix data, AbstractRealMatrix copy) {
		// Must iter by column
		for(int col = 0; col < data.getColumnDimension(); col++) {
			final double[] v = data.getColumn(col);
			final double mean = VecUtils.mean(v);
			final double sd = VecUtils.stdDev(v, mean);
			
			for(int row = 0; row < data.getRowDimension(); row++) {
				final double new_val = (v[row] - mean) / sd;
				copy.setEntry(row, col, new_val);
			}
		}
		
		return copy;
	}
	
	private void flagWarning() {
		hasWarnings = true;
	}

	
	/**
	 * Copies the underlying AbstractRealMatrix datastructure
	 * and returns the clone so as to prevent accidental referential
	 * alterations of the data.
	 * @return copy of data
	 */
	public AbstractRealMatrix getData() {
		return (AbstractRealMatrix) data.copy();
	}
	
	
	/**
	 * Returns the separability metric used to assess vector similarity/distance
	 * @return distance metric
	 */
	public GeometricallySeparable getSeparabilityMetric() {
		return dist;
	}
	
	
	/**
	 * Get the current seed being used for random state
	 * @return the random seed
	 */
	public Random getSeed() {
		return seed;
	}
	
	/**
	 * Whether the algorithm resulted in any warnings
	 * @return whether the clustering effort has generated any warnings
	 */
	public boolean hasWarnings() {
		return hasWarnings;
	}
	
	
	/**
	 * Get the model key, the model's unique UUID
	 * @return the model's unique UUID
	 */
	public UUID getKey() {
		return modelKey;
	}
	
	
	/**
	 * Get the state of the model's verbosity
	 * @return is the model set to verbose mode or not?
	 */
	public boolean getVerbose() {
		return verbose;
	}
	
	
	
	/** 
	 * Fit the model.
	 * In order to coalesce with the milieu of clust4j,
	 * the execution of this method should be synchronized on 'this'
	 */
	abstract public AbstractClusterer fit();
	
	
	
	/* -- LOGGER METHODS --  */
	@Override public void error(String msg) {
		if(verbose) Log.err(getLoggerTag(), msg);
	}
	
	@Override public void warn(String msg) {
		flagWarning();
		if(verbose) Log.warn(getLoggerTag(), msg);
	}
	
	@Override public void info(String msg) {
		if(verbose) Log.info(getLoggerTag(), msg);
	}
	
	@Override public void trace(String msg) {
		if(verbose) Log.trace(getLoggerTag(), msg);
	}
	
	@Override public void debug(String msg) {
		if(verbose) Log.debug(getLoggerTag(), msg);
	}
	
	/**
	 * Log info related to the internal state 
	 * of the model (not progress)
	 * @param msg
	 */
	public void meta(final String msg) {
		info("[meta "+getName()+"] " + msg);
	}
	
	/**
	 * Load a model from a FileInputStream
	 * @param fos
	 * @return
	 * @throws IOException
	 * @throws ClassNotFoundException
	 */
	public static AbstractClusterer loadModel(final FileInputStream fis) throws IOException, ClassNotFoundException {
		ObjectInputStream in = new ObjectInputStream(fis);
        AbstractClusterer ac = (AbstractClusterer) in.readObject();
        in.close();
        fis.close();
        
        return ac;
	}
	
	/**
	 * Save a model to FileOutputStream
	 * @param fos
	 * @throws IOException
	 */
	public void saveModel(final FileOutputStream fos) throws IOException {
		ObjectOutputStream out = new ObjectOutputStream(fos);
		out.writeObject(this);
		out.close();
		fos.close();
	}
	
	protected void setSeparabilityMetric(final GeometricallySeparable sep) {
		this.dist = sep;
	}
}

package com.clust4j.algo;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.RejectedExecutionException;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.GlobalState;
import com.clust4j.algo.preprocess.FeatureNormalization;
import com.clust4j.kernel.Kernel;
import com.clust4j.log.Log;
import com.clust4j.log.LogTimeFormatter;
import com.clust4j.log.LogTimer;
import com.clust4j.log.Loggable;
import com.clust4j.utils.DeepCloneable;
import com.clust4j.utils.Distance;
import com.clust4j.utils.DistanceMetric;
import com.clust4j.utils.GeometricallySeparable;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.NaNException;
import com.clust4j.utils.Named;
import com.clust4j.utils.SimilarityMetric;

import static com.clust4j.GlobalState.ParallelismConf.ALLOW_AUTO_PARALLELISM;

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
public abstract class AbstractClusterer 
		extends BaseModel 
		implements Loggable, Named, java.io.Serializable {
	
	private static final long serialVersionUID = -3623527903903305017L;
	
	
	/** The default {@link FeatureNormalization} enum to use. 
	 *  The default is {@link FeatureNormalization#STANDARD_SCALE} */
	public static FeatureNormalization DEF_NORMALIZER = FeatureNormalization.STANDARD_SCALE;
	
	/** Whether algorithms should by default behave in a verbose manner */
	public static boolean DEF_VERBOSE = false;
	
	/** Whether algorithms should by default normalize the columns */
	public static boolean DEF_SCALE = false;
	
	/** By default, uses the {@link GlobalState#DEFAULT_RANDOM_STATE} */
	final static protected Random DEF_SEED = GlobalState.DEFAULT_RANDOM_STATE;
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
	abstract public static class BaseClustererPlanner 
			implements DeepCloneable, BaseClassifierPlanner {
		
		//abstract public AbstractClusterer buildNewModelInstance(final AbstractRealMatrix data);
		@Override abstract public BaseClustererPlanner copy();
		abstract public FeatureNormalization getNormalizer();
		abstract public GeometricallySeparable getSep();
		abstract public boolean getScale();
		abstract public Random getSeed();
		abstract public boolean getVerbose();
		abstract public BaseClustererPlanner setNormalizer(final FeatureNormalization norm);
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
		handleData(data);
		
		// Log info
		info("initializing " + getName() + 
				" clustering with " + data.getRowDimension() + 
				" x " + data.getColumnDimension() + " data matrix");
		
		if(this.dist instanceof Kernel) {
			warn("running " + getName() + " in Kernel mode can be an expensive option");
		}
		
		
		String clzNm = "AbstractClusterer";
		meta("model key="+modelKey, clzNm);
		meta((similarity ? "similarity" : "distance") + 
				" metric=" + dist.getName(), clzNm);
		meta("scale="+planner.getScale(), clzNm);
		meta("force_parallelism="+GlobalState.ParallelismConf.FORCE_PARALLELISM_WHERE_POSSIBLE, clzNm);
		meta("allow_auto_parallelism="+GlobalState.ParallelismConf.ALLOW_AUTO_PARALLELISM, clzNm);
		
		
		// Scale if needed
		if(!planner.getScale()) {
			this.data = (AbstractRealMatrix) data.copy();
			metaWarn("feature normalization option is set to false; this is discouraged");
		} else {
			meta("normalizing matrix columns", planner.getNormalizer().toString());
			this.data = planner.getNormalizer().operate(data);
		}
	} // End constructor
	
	
	
	
	final private void handleData(final AbstractRealMatrix data) {
		if(data.getRowDimension() == 0)
			throw new IllegalArgumentException("empty data");
		
		
		// Check for nans in the matrix either serially or in parallel
		boolean containsNan = false;
		if(!ALLOW_AUTO_PARALLELISM) {
			info("checking input data for NaNs serially");
			containsNan = MatUtils.containsNaN(data);
		} else {
			try { // Try distributed job
				info("checking input data for NaNs using core-distributed task");
				containsNan = MatUtils.containsNaNDistributed(data);
			} catch(RejectedExecutionException | OutOfMemoryError e) { // can't schedule parallel job/HS error
				warn("parallel NaN check failed, reverting to serial check");
				containsNan = MatUtils.containsNaN(data);
			}
		}
		
		
		if(containsNan) {
			String error = "NaN in input data. Select a matrix imputation method for incomplete records";
			error(error);
			throw new NaNException(error);
		}
	}
	
	
	/**
	 * A model must have the same key, data and class name
	 * in order to equal another model
	 */
	@Override
	public boolean equals(Object o) {
		if(this == o)
			return true;
		if(o instanceof AbstractClusterer) {
			AbstractClusterer a = (AbstractClusterer)o;
			if(!this.getKey().equals(a.getKey()))
				return false;
			
			return this.data.equals(a.data)
				&& this.getClass().equals(a.getClass());
		}
		
		return false;
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
	
	@Override
	public int hashCode() {
		int result = 17;
		return result 
			^ (verbose ? 1 : 0)
			^ (getKey().hashCode())
			^ (dist instanceof DistanceMetric ? 31 :
				dist instanceof SimilarityMetric ? 53 : 1)
			^ (hasWarnings ? 1 : 0)
			^ seed.hashCode();
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
	 * Fits the model. In order to fit the style of clust4j,
	 * the execution of this method should be synchronized on 'this'. This
	 * is due to the volatile nature of many of the instance class variables.
	 */
	@Override abstract public AbstractClusterer fit();
	
	
	
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
	
	@Override public void wallInfo(LogTimer timer, String info) {
		if(verbose) info(timer.wallMsg() + info);
	}
	
	/**
	 * Write the time the algorithm took to complete
	 * @param start
	 */
	@Override public void sayBye(final LogTimer timer) {
		final long start = timer._start;
		wallInfo(timer, "model "+getKey()+" completed in " + 
			LogTimeFormatter.millis(System.currentTimeMillis()-start, false) + 
			System.lineSeparator());
	}
	
	/**
	 * Log info related to the internal state 
	 * of the model (not progress)
	 * @param msg
	 */
	void meta(final String msg) {
		meta(msg, getName());
	}
	
	void meta(final String msg, final String nm) {
		info("[meta "+nm+"] " + msg);
	}
	
	void metaWarn(final String msg) {
		metaWarn(msg, getName());
	}
	
	void metaWarn(final String msg, final String nm) {
		warn("[meta "+nm+"] " + msg);
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

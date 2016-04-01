package com.clust4j.algo;

import java.text.NumberFormat;
import java.util.Random;
import java.util.UUID;

import lombok.NonNull;

import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;

import com.clust4j.Clust4j;
import com.clust4j.GlobalState;
import com.clust4j.NamedEntity;
import com.clust4j.algo.preprocess.FeatureNormalization;
import com.clust4j.except.NaNException;
import com.clust4j.kernel.Kernel;
import com.clust4j.log.Log;
import com.clust4j.log.LogTimer;
import com.clust4j.log.Loggable;
import com.clust4j.metrics.pairwise.Distance;
import com.clust4j.metrics.pairwise.DistanceMetric;
import com.clust4j.metrics.pairwise.GeometricallySeparable;
import com.clust4j.metrics.pairwise.SimilarityMetric;
import com.clust4j.utils.DeepCloneable;
import com.clust4j.utils.TableFormatter;
import com.clust4j.utils.TableFormatter.Table;

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
		implements Loggable, NamedEntity, java.io.Serializable {
	
	private static final long serialVersionUID = -3623527903903305017L;
	final static TableFormatter formatter;
	
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
	final protected Array2DRowRealMatrix data;
	/** Similarity metric */
	protected GeometricallySeparable dist_metric;
	/** Seed for any shuffles */
	protected final Random random_state;
	/** Verbose for heavily logging */
	final private boolean verbose;
	/** Whether we scale or not */
	final boolean normalized;
	/** Whether to use parallelism */
	final boolean parallel;
	/** The normalizer */
	final FeatureNormalization normalizer;
	
	
	
	/** Have any warnings occurred -- volatile because can change */
	volatile private boolean hasWarnings = false;
	final ModelSummary fitSummary;
	

	
	
	/**
	 * Base planner class many clustering algorithms
	 * will extend with static inner classes. Some clustering
	 * algorithms will require more parameters and must provide
	 * the interface for the getting/setting of such parameters.
	 * 
	 * @author Taylor G Smith
	 */
	abstract public static class BaseClustererPlanner 
			extends Clust4j // So all are serializable
			implements DeepCloneable, BaseClassifierPlanner {
		private static final long serialVersionUID = -5830795881133834268L;
		
		@Override abstract public BaseClustererPlanner copy();
		abstract public FeatureNormalization getNormalizer();
		abstract public boolean getParallel();
		abstract public GeometricallySeparable getSep();
		abstract public boolean getScale();
		abstract public Random getSeed();
		abstract public boolean getVerbose();
		abstract public BaseClustererPlanner setNormalizer(final FeatureNormalization norm);
		abstract public BaseClustererPlanner setScale(final boolean b);
		abstract public BaseClustererPlanner setSeed(final Random rand);
		abstract public BaseClustererPlanner setVerbose(final boolean b);
		abstract public BaseClustererPlanner setSep(final GeometricallySeparable dist);
		abstract public BaseClustererPlanner setForceParallel(final boolean b);
	}
	
	
	
	// Initializers
	static {
		NumberFormat nf = NumberFormat.getInstance(TableFormatter.DEFAULT_LOCALE);
		nf.setMaximumFractionDigits(5);
		formatter = new TableFormatter(nf);
		formatter.leadWithEmpty = false;
		formatter.setWhiteSpace(1);
	}
	
	/**
	 * Build a new instance from another caller
	 * @param caller
	 */
	protected AbstractClusterer(AbstractClusterer caller) {
		this(caller, null);
	}
	
	/**
	 * Internal constructor giving precedence to the planning class if not null
	 * @param caller
	 * @param planner
	 */
	protected AbstractClusterer(AbstractClusterer caller, BaseClustererPlanner planner) {
		this.dist_metric= null == planner ? caller.dist_metric : planner.getSep();
		this.verbose 	= null == planner ? false : planner.getVerbose(); // if another caller, default to false
		this.modelKey 	= UUID.randomUUID();
		this.random_state 		= null == planner ? caller.random_state : planner.getSeed();
		this.data 		= caller.data; // Use the reference
		this.normalized	= caller.normalized;
		this.parallel 	= caller.parallel;
		this.fitSummary = new ModelSummary(getModelFitSummaryHeaders());
		this.normalizer = null == planner ? caller.normalizer : planner.getNormalizer();
	}
	
	protected AbstractClusterer(AbstractRealMatrix data, BaseClustererPlanner planner, boolean as_is) {
		
		this.dist_metric = planner.getSep();
		this.verbose = planner.getVerbose();
		this.modelKey = UUID.randomUUID();
		this.random_state = planner.getSeed();

		// Scale if needed
		this.normalized = planner.getScale();
		this.normalizer = planner.getNormalizer();
		
		// Determine whether we should parallelize
		this.parallel = planner.getParallel()
			|| (GlobalState.ParallelismConf.ALLOW_AUTO_PARALLELISM
			&& (data.getRowDimension() * data.getColumnDimension()) 
			> GlobalState.ParallelismConf.MIN_ELEMENTS);
		
		if(this.dist_metric instanceof Kernel)
			warn("running " + getName() + " in Kernel mode can be an expensive option");
		
		// Handle data, now...
		this.data = as_is ? 
			(Array2DRowRealMatrix)data : // internally, always 2d...
				initData(data);
			
		this.fitSummary = new ModelSummary(getModelFitSummaryHeaders());
	}
	
	/**
	 * Base clusterer constructor. Sets up the distance measure,
	 * and if necessary scales data.
	 * @param data
	 * @param planner
	 */
	public AbstractClusterer(@NonNull AbstractRealMatrix data, @NonNull BaseClustererPlanner planner) {
		this(data, planner, false);
	}
	
	
	
	final private Array2DRowRealMatrix initData(final AbstractRealMatrix data) {
		final int m = data.getRowDimension(), n = data.getColumnDimension();
		final double[][] ref = new double[m][n];
		
		double entry;
		for(int i = 0; i < m; i++) {
			for(int j = 0; j < n; j++) {
				entry = data.getEntry(i, j);
						
				if(Double.isNaN(entry)) {
					String error = "NaN in input data. Select a matrix imputation method for incomplete records";
					error(error);
					throw new NaNException(error);
				}
				
				ref[i][j] = entry;
			}
		}
		
		if(!normalized)
			warn("feature normalization option is set to false; this is discouraged");
		
		return new Array2DRowRealMatrix(
			normalized ? normalizer.operate(ref) : ref,
		false);
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
				&& this.getClass().equals(a.getClass())
				//&& this.hashCode() == a.hashCode()
				;
		}
		
		return false;
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
		return dist_metric;
	}
	
	
	/**
	 * Get the current seed being used for random state
	 * @return the random seed
	 */
	public Random getSeed() {
		return random_state;
	}
	
	/**
	 * Whether the algorithm resulted in any warnings
	 * @return whether the clustering effort has generated any warnings
	 */
	@Override
	public boolean hasWarnings() {
		return hasWarnings;
	}
	
	@Override
	public int hashCode() {
		int result = 17;
		return result 
			^ (verbose ? 1 : 0)
			^ (getKey().hashCode())
			^ (dist_metric instanceof DistanceMetric ? 31 :
				dist_metric instanceof SimilarityMetric ? 53 : 1)
			// ^ (hasWarnings ? 1 : 0) // removed because forces state dependency
			^ random_state.hashCode();
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
	
	
	/* -- LOGGER METHODS --  */
	@Override public void error(String msg) {
		if(verbose) Log.err(getLoggerTag(), msg);
	}
	
	@Override public void warn(String msg) {
		hasWarnings = true;
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
	 * Write the time the algorithm took to complete
	 * @param timer
	 */
	@Override public void sayBye(final LogTimer timer) {
		logFitSummary();
		info("model "+getKey()+" fit completed in " + timer.toString());
	}
	
	/**
	 * Used for logging the initialization summary.
	 */
	private void logFitSummary() {
		info("--");
		info("Model Fit Summary:");
		final Table tab = formatter.format(fitSummary);
		final String fmt = tab.toString();
		final String sep = System.getProperty("line.separator");
		final String[] summary = fmt.split(sep);
		
		// Sometimes the fit summary can be overwhelmingly long..
		// Only want to show top few & bottom few. (extra 1 on top for header)
		final int top = 6, bottom = top - 1;
		int topThresh = top, bottomThresh;
		if(summary.length > top + bottom) {
			// calculate the bottom thresh
			bottomThresh = summary.length - bottom;
		} else {
			topThresh = summary.length;
			bottomThresh = 0;
		}
		
		
		int iter = 0;
		boolean shownBreak = false;
		for(String line: summary) {
			if(iter < topThresh || iter > bottomThresh)
				info(line);
			else if(!shownBreak) {
				// first after top thresh
				info(tab.getTableBreak());
				shownBreak = true;
			}
			
			iter++;
		}
	}
	
	/**
	 * Used for logging the initialization summary
	 */
	final void logModelSummary() {
		info("--");
		info("Model Init Summary:");
		final String sep = System.getProperty("line.separator");
		
		final String[] summary = formatter
			.format(modelSummary())
			.toString()
			.split(sep);
		
		for(String line: summary)
			info(line);
	}
	
	protected void setSeparabilityMetric(final GeometricallySeparable sep) {
		this.dist_metric = sep;
	}
	
	
	

	/** 
	 * Fits the model
	 */
	@Override
	abstract public AbstractClusterer fit();
	
	protected abstract ModelSummary modelSummary();
	protected abstract Object[] getModelFitSummaryHeaders();
}

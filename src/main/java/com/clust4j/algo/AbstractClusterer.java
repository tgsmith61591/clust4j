/*******************************************************************************
 *    Copyright 2015, 2016 Taylor G Smith
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *******************************************************************************/

package com.clust4j.algo;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.Random;
import java.util.UUID;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.GlobalState;
import com.clust4j.NamedEntity;
import com.clust4j.except.ModelNotFitException;
import com.clust4j.except.NaNException;
import com.clust4j.kernel.Kernel;
import com.clust4j.log.Log;
import com.clust4j.log.LogTimer;
import com.clust4j.log.Loggable;
import com.clust4j.metrics.pairwise.Distance;
import com.clust4j.metrics.pairwise.DistanceMetric;
import com.clust4j.metrics.pairwise.GeometricallySeparable;
import com.clust4j.metrics.pairwise.SimilarityMetric;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.TableFormatter.Table;
import com.clust4j.utils.VecUtils;

/**
 * 
 * The highest level of cluster abstraction in clust4j, AbstractClusterer
 * provides the interface for classifier clustering (both supervised and unsupervised).
 * It also provides all the functionality for any BaseClustererPlanner classes and logging.
 * 
 * @author Taylor G Smith &lt;tgsmith61591@gmail.com&gt;
 *
 */
public abstract class AbstractClusterer 
		extends BaseModel 
		implements Loggable, NamedEntity, java.io.Serializable, MetricValidator {
	
	private static final long serialVersionUID = -3623527903903305017L;
	
	/** Whether algorithms should by default behave in a verbose manner */
	public static boolean DEF_VERBOSE = false;
	
	/** By default, uses the {@link GlobalState#DEFAULT_RANDOM_STATE} */
	protected final static Random DEF_SEED = GlobalState.DEFAULT_RANDOM_STATE;
	final public static GeometricallySeparable DEF_DIST = Distance.EUCLIDEAN;
	/** The model id */
	final private String modelKey;
	
	
	
	/** Underlying data */
	final protected Array2DRowRealMatrix data;
	/** Similarity metric */
	protected GeometricallySeparable dist_metric;
	/** Seed for any shuffles */
	protected final Random random_state;
	/** Verbose for heavily logging */
	final private boolean verbose;
	/** Whether to use parallelism */
	protected final boolean parallel;
	/** Whether the entire matrix is comprised of only one unique value */
	protected boolean singular_value;
	
	
	
	/** Have any warnings occurred -- volatile because can change */
	volatile private boolean hasWarnings = false;
	final private ArrayList<String> warnings = new ArrayList<>();
	protected final ModelSummary fitSummary;
	
	
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
	protected AbstractClusterer(AbstractClusterer caller, BaseClustererParameters planner) {
		this.dist_metric	= null == planner ? caller.dist_metric : planner.getMetric();
		this.verbose 		= null == planner ? false : planner.getVerbose(); // if another caller, default to false
		this.modelKey 		= getName() + "_" + UUID.randomUUID();
		this.random_state 	= null == planner ? caller.random_state : planner.getSeed();
		this.data 			= caller.data; // Use the reference
		this.parallel 		= caller.parallel;
		this.fitSummary 	= new ModelSummary(getModelFitSummaryHeaders());
		this.singular_value = caller.singular_value;
	}
	
	protected AbstractClusterer(RealMatrix data, BaseClustererParameters planner, boolean as_is) {
		
		this.dist_metric = planner.getMetric();
		this.verbose = planner.getVerbose();
		this.modelKey = getName() + "_" + UUID.randomUUID();
		this.random_state = planner.getSeed();
		
		// Determine whether we should parallelize
		this.parallel = planner.getParallel() && GlobalState.ParallelismConf.PARALLELISM_ALLOWED;
		
		/*
		 * If user tried to force serial, but we just can't...
		 */
		if(!parallel && planner.getParallel())
			info("min num cores required for parallel: " + GlobalState.ParallelismConf.MIN_CORES_REQUIRED);
		
		if(this.dist_metric instanceof Kernel)
			warn("running " + getName() + " in Kernel mode can be an expensive option");
		
		// Handle data, now...
		this.data = as_is ? 
			(Array2DRowRealMatrix)data : // internally, always 2d...
				initData(data);
		if(singular_value)
			warn("all elements in input matrix are equal ("+data.getEntry(0, 0)+")");
			
		this.fitSummary = new ModelSummary(getModelFitSummaryHeaders());
	}
	
	/**
	 * Base clusterer constructor. Sets up the distance measure,
	 * and if necessary scales data.
	 * @param data
	 * @param planner
	 */
	protected AbstractClusterer(RealMatrix data, BaseClustererParameters planner) {
		this(data, planner, false);
	}
	
	
	
	final private Array2DRowRealMatrix initData(final RealMatrix data) {
		final int m = data.getRowDimension(), n = data.getColumnDimension();
		final double[][] ref = new double[m][n];
		final HashSet<Double> unique = new HashSet<>();
		
		// Used to compute variance on the fly for summaries later...
		double[] sum   = new double[n];
		double[] sumSq = new double[n];
		double[] maxes = VecUtils.rep(Double.NEGATIVE_INFINITY, n);
		double[] mins  = VecUtils.rep(Double.POSITIVE_INFINITY, n);
		
		// This will store summaries for each column + a header
		ModelSummary summaries = new ModelSummary(new Object[]{
			"Feature #","Variance","Std. Dev","Mean","Max","Min"
		});
		
		/*
		 * Internally performs the copy
		 */
		double entry;
		for(int i = 0; i < m; i++) {
			for(int j = 0; j < n; j++) {
				entry = data.getEntry(i, j);
						
				if(Double.isNaN(entry)) {
					error(new NaNException("NaN in input data. "
						+ "Select a matrix imputation method for "
						+ "incomplete records"));
				} else {
					// copy the entry
					ref[i][j] = entry;
					unique.add(entry);
					
					// capture stats...
					sumSq[j] += entry * entry;
					sum[j]   += entry;
					maxes[j]  = FastMath.max(entry, maxes[j]);
					mins[j]   = FastMath.min(entry, mins[j]);
					
					// if it's the last row, we can compute these:
					if(i == m - 1) {
						double var = (sumSq[j] - (sum[j]*sum[j])/(double)m ) / ((double)m - 1.0);
						if(var == 0) {
							warn("zero variance in feature " + j);
						}
						
						summaries.add(new Object[]{
							j, // feature num
							var, // var
							m < 2 ? Double.NaN : FastMath.sqrt(var), // std dev
							sum[j] / (double)m, // mean
							maxes[j], // max
							mins[j] // min
						});
					}
				}
			}
		}
		
		// Log the summaries
		summaryLogger(formatter.format(summaries));
		
		if(unique.size() == 1)
			this.singular_value = true;

		/*
		 * Don't need to copy again, because already internally copied...
		 */
		return new Array2DRowRealMatrix(ref, false);
	}
	
	
	/**
	 * A model must have the same key, data and class name
	 * in order to equal another model. It is extremely unlikely
	 * that a model will share a UUID with another. In fact, the probability 
	 * of one duplicate would be about 50% if every person on 
	 * Earth owned 600 million UUIDs.
	 */
	@Override
	public boolean equals(Object o) {
		if(this == o)
			return true;
		if(o instanceof AbstractClusterer) {
			AbstractClusterer a = (AbstractClusterer)o;
			if(!this.getKey().equals(a.getKey()))
				return false;
			
			return MatUtils.equalsExactly(this.data.getDataRef(), a.data.getDataRef())
				&& this.getClass().equals(a.getClass())
				//&& this.hashCode() == a.hashCode()
				;
		}
		
		return false;
	}
	
	/**
	 * Handles all label copies and ModelNotFitExceptions. 
	 * <b>This should be called within <tt>getLabels()</tt> operations</b>
	 * @param data
	 * @param shuffleOrder
	 * @return
	 */
	protected int[] handleLabelCopy(int[] labels) throws ModelNotFitException {
		if(null == labels) {
			error(new ModelNotFitException("model has not been fit yet"));
			return null;
		} else {
			return VecUtils.copy(labels);
		}
	}
	
	/**
	 * Copies the underlying AbstractRealMatrix datastructure
	 * and returns the clone so as to prevent accidental referential
	 * alterations of the data.
	 * @return copy of data
	 */
	public RealMatrix getData() {
		return data.copy();
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
			^ random_state.hashCode()
			^ data.hashCode();
	}
	
	
	/**
	 * Get the model key, the model's unique UUID
	 * @return the model's unique UUID
	 */
	public String getKey() {
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
	 * Returns a collection of warnings if there are any, otherwise null
	 * @return
	 */
	final public Collection<String> getWarnings() {
		return warnings.isEmpty() ? null : warnings;
	}
	
	
	/* -- LOGGER METHODS --  */
	@Override public void error(String msg) {
		if(verbose) Log.err(getLoggerTag(), msg);
	}
	
	@Override public void error(RuntimeException thrown) {
		error(thrown.getMessage());
		throw thrown;
	}
	
	@Override public void warn(String msg) {
		hasWarnings = true;
		warnings.add(msg);
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
		summaryLogger(tab);
	}
	
	/**
	 * Used for logging the initialization summary
	 */
	protected final void logModelSummary() {
		info("--");
		info("Model Init Summary:");
		final Table tab = formatter.format(modelSummary());
		summaryLogger(tab);
	}
	
	/**
	 * Handles logging of tables
	 */
	final private void summaryLogger(Table tab) {
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
	
	protected void setSeparabilityMetric(final GeometricallySeparable sep) {
		this.dist_metric = sep;
	}
	
	

	/** 
	 * Fits the model
	 */
	@Override abstract protected AbstractClusterer fit();
	protected abstract ModelSummary modelSummary();
	protected abstract Object[] getModelFitSummaryHeaders();
}

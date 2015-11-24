package com.clust4j.algo;

import java.util.ArrayList;
import java.util.Random;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.kernel.RadialBasisKernel;
import com.clust4j.kernel.GaussianKernel;
import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.utils.CentroidLearner;
import com.clust4j.utils.Classifier;
import com.clust4j.utils.Convergeable;
import com.clust4j.utils.GeometricallySeparable;
import com.clust4j.utils.VecUtils;


/**
 * Mean shift is a procedure for locating the maxima of a density function given discrete 
 * data sampled from that function. It is useful for detecting the modes of this density. 
 * This is an iterative method, and we start with an initial estimate <i>x</i> . Let a
 * {@link RadialBasisKernel} function be given. This function determines the weight of nearby 
 * points for re-estimation of the mean. Typically a {@link GaussianKernel} kernel on the 
 * distance to the current estimate is used.
 * 
 * @see <a href="https://en.wikipedia.org/wiki/Mean_shift">Mean shift on Wikipedia</a>
 * @author Taylor G Smith
 */
public class MeanShift 
		extends AbstractDensityClusterer 
		implements CentroidLearner, Classifier, Convergeable {
	final public static int DEF_MAX_ITER = 300;
	final public static double DEF_MIN_CHANGE = 0d;
	final public static int DEF_MIN_BIN_FREQ = 1;
	
	
	
	/** Track convergence */
	private boolean converged = false;
	
	/** Count iterations */
	private int itersElapsed = 0;
	
	/** The max iterations */
	private final int maxIter;
	
	private final int minBinFreq;
	
	/** Min change convergence criteria */
	private final double minChange;
	
	/** The kernel bandwidth */
	private final double bandwidth;
	
	/** Which kind of kernel to generate for each centroid */
	private final Class<? extends RadialBasisKernel> rbfKernelClass;
	
	/** The kernel to use */
	private final RadialBasisKernel kernel;

	/** Class labels */
	private int[] labels;
	
	/** The centroid records */
	private ArrayList<double[]> centroids = new ArrayList<double[]>();
	
	
	
	
	
	/**
	 * Default constructor
	 * @param data
	 * @param bandwidth
	 */
	public MeanShift(AbstractRealMatrix data, final double bandwidth) {
		this(data, new MeanShiftPlanner(bandwidth));
	}
	
	/**
	 * Constructor with custom MeanShiftPlanner
	 * @param data
	 * @param planner
	 */
	public MeanShift(AbstractRealMatrix data, MeanShiftPlanner planner) {
		super(data, planner);
		
		if(planner.bandwidth <= 0.0)
			throw new IllegalArgumentException("bandwidth must be greater than 0.0");
		
		this.bandwidth = planner.bandwidth;
		this.maxIter = planner.maxIter;
		this.minBinFreq = planner.minBinFreq;
		this.minChange = planner.minChange;
		this.rbfKernelClass = planner.rbfKernelClass;
		
		
		if(verbose) {
			meta("bandwidth="+bandwidth);
			meta("maxIter="+maxIter);
			meta("minBinFreq="+minBinFreq);
			meta("minChange="+minChange);
			
			final String[] kclz = rbfKernelClass.toString().split("\\.");
			meta("kernel class="+ (kclz.length>1?kclz[kclz.length-1]:kclz[0]) );
		}
		
		
		kernel = initKernel(bandwidth);
	}
	
	
	
	
	
	/**
	 * A builder class to provide an easier constructing
	 * interface to set custom parameters for DBSCAN
	 * @author Taylor G Smith
	 */
	final public static class MeanShiftPlanner extends AbstractClusterer.BaseClustererPlanner {
		private double bandwidth;
		private int maxIter = DEF_MAX_ITER;
		private int minBinFreq = DEF_MIN_BIN_FREQ;
		private double minChange = DEF_MIN_CHANGE;
		private boolean scale = DEF_SCALE;
		private Random seed = DEF_SEED;
		private GeometricallySeparable dist	= DEF_DIST;
		private boolean verbose	= DEF_VERBOSE;
		private Class<? extends RadialBasisKernel> rbfKernelClass = RadialBasisKernel.class;
		
		public MeanShiftPlanner(final double bandwidth) {
			this.bandwidth = bandwidth;
		}
		
		@Override
		public GeometricallySeparable getSep() {
			return dist;
		}
		
		@Override
		public boolean getScale() {
			return scale;
		}
		
		@Override
		public Random getSeed() {
			return seed;
		}
		
		@Override
		public boolean getVerbose() {
			return verbose;
		}
		
		public MeanShiftPlanner setMaxIter(final int max) {
			this.maxIter = max;
			return this;
		}
		
		public MeanShiftPlanner setMinBinFreq(final int min) {
			this.minBinFreq = min;
			return this;
		}
		
		public MeanShiftPlanner setMinChange(final double min) {
			this.minChange = min;
			return this;
		}
		
		public MeanShiftPlanner setRbfKernelClass(final Class<? extends RadialBasisKernel> clazz) {
			this.rbfKernelClass = clazz;
			return this;
		}
		
		@Override
		public MeanShiftPlanner setScale(final boolean scale) {
			this.scale = scale;
			return this;
		}
		
		@Override
		public MeanShiftPlanner setSeed(final Random seed) {
			this.seed = seed;
			return this;
		}
		
		@Override
		public MeanShiftPlanner setSep(final GeometricallySeparable dist) {
			this.dist = dist;
			return this;
		}
		
		@Override
		public MeanShiftPlanner setVerbose(final boolean v) {
			this.verbose = v;
			return this;
		}
	}
	
	
	public double getBandwidth() {
		return bandwidth;
	}

	public RadialBasisKernel getKernel() {
		return kernel;
	}
	
	/**
	 * Try to instantiate a kernel from the provided class
	 * @param b
	 * @return
	 */
	private RadialBasisKernel initKernel(final double b) {
		String es;
		
		try {
			final RadialBasisKernel rbf = rbfKernelClass.newInstance();
			rbf.setSigma(b);
			return rbf;
		} catch(InstantiationException e) {
			es = e.getMessage();
			if(verbose) error(es);
		} catch(IllegalAccessException e) {
			es = e.getMessage();
			if(verbose) error(es);
		}
		
		throw new InternalError("unable to instantiate kernel: " + es);
	}
	
	@Override
	public boolean didConverge() {
		return converged;
	}
	
	@Override
	public int itersElapsed() {
		return itersElapsed;
	}
	
	public int getMaxIter() {
		return maxIter;
	}
	
	public int getMinBinFreq() {
		return minBinFreq;
	}
	
	public double getMinChange() {
		return minChange;
	}

	@Override
	public String getName() {
		return "MeanShift";
	}


	@Override
	public Algo getLoggerTag() {
		return com.clust4j.log.Log.Tag.Algo.MEANSHIFT;
	}


	@Override
	public MeanShift fit() {
		synchronized(this) { // Synch because isTrained is a race condition
			
			if(null!=labels) // Already fit this model
				return this;
			
			
			final int m = data.getRowDimension(), n = data.getColumnDimension();
			if(verbose) info("identifying neighborhoods within bandwidth");
			
			
			
			
			// TODO:
			
			return this;
		} // End synch
		
	} // End train


	@Override
	public ArrayList<double[]> getCentroids() {
		final ArrayList<double[]> cent = new ArrayList<double[]>();
		for(double[] d : centroids)
			cent.add(VecUtils.copy(d));
		
		return cent;
	}

	@Override
	public int[] getLabels() {
		return labels;
	}
}

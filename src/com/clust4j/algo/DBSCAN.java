package com.clust4j.algo;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.utils.GeometricallySeparable;
import com.clust4j.utils.Classifier;

import static com.clust4j.algo.DBSCAN.Method.*;

public class DBSCAN extends AbstractDensityClusterer implements Classifier {
	/**
	 * The different types of methods to 
	 * use in the clustering process.
	 * @author Taylor G Smith */
	public static enum Method {
		HYBRID,
		RAW,
		DIST
	}
	
	
	final public static Method DEF_METHOD = DIST;
	final public static int DEF_MIN_PTS = 5;
	
	final private int minPts;
	final private double eps;
	final private Method method;
	
	private boolean isTrained = false;
	private int[] labels = null;
	
	
	/**
	 * A builder class to provide an easier constructing
	 * interface to set custom parameters for DBSCAN
	 * @author Taylor G Smith
	 */
	final public static class DBSCANPlanner extends AbstractClusterer.BaseClustererPlanner {
		private double eps;
		private int minPts		= DEF_MIN_PTS;
		private boolean scale	= DEF_SCALE;
		private GeometricallySeparable dist	= DEF_DIST;
		private Method method	= DEF_METHOD;
		
		public DBSCANPlanner(final double eps) {
			this.eps = eps;
		}
		
		@Override
		public GeometricallySeparable getDist() {
			return dist;
		}
		
		@Override
		public boolean getScale() {
			return scale;
		}
		
		public DBSCANPlanner setMinPts(final int minPts) {
			this.minPts = minPts;
			return this;
		}
		
		@Override
		public DBSCANPlanner setScale(final boolean scale) {
			this.scale = scale;
			return this;
		}
		
		@Override
		public DBSCANPlanner setDist(final GeometricallySeparable dist) {
			this.dist = dist;
			return this;
		}
		
		public DBSCANPlanner setMethod(final Method method) {
			this.method = method;
			return this;
		}
	}
	
	
	
	/**
	 * Constructs an instance of DBSCAN from the default values
	 * @param eps
	 * @param data
	 */
	public DBSCAN(final AbstractRealMatrix data, final double eps) {
		this(data, new DBSCANPlanner(eps));
	}
	
	/**
	 * Constructs an instance of DBSCAN from the provided builder
	 * @param builder
	 * @param data
	 */
	public DBSCAN(final AbstractRealMatrix data, final DBSCANPlanner builder) {
		super(data, builder);
		
		this.minPts = builder.minPts;
		this.eps 	= builder.eps;
		this.method	= builder.method;
	}
	

	
	public double getEps() {
		return eps;
	}
	
	@Override
	public int[] getPredictedLabels() {
		return labels;
	}
	
	public Method getMethod() {
		return method;
	}
	
	public int getMinPts() {
		return minPts;
	}
	
	@Override
	public String getName() {
		return "DBSCAN";
	}
	
	@Override
	public boolean isTrained() {
		return isTrained;
	}
	
	@Override
	public int predict(final double[] newRecord) {
		// TODO:
		return 0;
	}
	
	@Override
	final public void train() {
		if(isTrained)
			return;
		
		// TODO:
		throw new UnsupportedOperationException("Not yet implemented");
	}
}

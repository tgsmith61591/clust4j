package com.clust4j.algo;

import java.util.ArrayList;
import java.util.Random;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.kernel.RadialBasisKernel;
import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.utils.CentroidLearner;
import com.clust4j.utils.GeometricallySeparable;

public class MeanShift extends AbstractDensityClusterer implements CentroidLearner {
	/**
	 * The kernel bandwidth
	 */
	private final double bandwidth;
	
	/**
	 * Which kind of kernel to generate for each centroid
	 */
	private final Class<? extends RadialBasisKernel> rbfKernelClass;

	
	
	
	
	public MeanShift(AbstractRealMatrix data, MeanShiftPlanner planner) {
		super(data, planner);
		
		if(planner.bandwidth <= 0.0)
			throw new IllegalArgumentException("bandwidth must be greater than 0.0");
		
		this.bandwidth = planner.bandwidth;
		this.rbfKernelClass = planner.rbfKernelClass;
	}

	
	
	
	
	/**
	 * A builder class to provide an easier constructing
	 * interface to set custom parameters for DBSCAN
	 * @author Taylor G Smith
	 */
	final public static class MeanShiftPlanner extends AbstractClusterer.BaseClustererPlanner {
		private double bandwidth;
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
		
		public MeanShiftPlanner setVerbose(final boolean v) {
			this.verbose = v;
			return this;
		}
	}
	
	
	public double getBandwidth() {
		return bandwidth;
	}

	public Class<? extends RadialBasisKernel> getKernelClass() {
		return rbfKernelClass;
	}
	
	private RadialBasisKernel initKernel(final double b) {
		try {
			final RadialBasisKernel rbf = rbfKernelClass.newInstance();
			rbf.setSigma(b);
			return rbf;
		} catch(InstantiationException e) {
			if(verbose) error(e.getMessage());
		} catch(IllegalAccessException e) {
			if(verbose) error(e.getMessage());
		}
		
		throw new InternalError("unable to instantiate kernel");
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
			
			// TODO:
			throw new UnsupportedOperationException("Not yet implemented");
		} // End synch
		
	} // End train


	@Override
	public ArrayList<double[]> getCentroids() {
		// TODO Auto-generated method stub
		return null;
	}
	
	
}

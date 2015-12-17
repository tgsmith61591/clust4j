package com.clust4j.algo.prep;

import java.util.Random;

import com.clust4j.algo.NearestNeighbors;
import com.clust4j.algo.NearestNeighbors.RunMode;
import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.utils.Distance;
import com.clust4j.utils.GeometricallySeparable;

public class NearestNeighborImputation extends MatrixImputation {
	final static public int DEF_K = NearestNeighbors.DEF_K;
	final static public double DEF_RADIUS = NearestNeighbors.DEF_EPS_RADIUS;
	final static public GeometricallySeparable DEF_METRIC = Distance.EUCLIDEAN;
	final static public RunMode DEF_RUN_MODE = RunMode.K_NEAREST;
	
	private int k = DEF_K;
	private double radius = DEF_RADIUS;
	private GeometricallySeparable sep = DEF_METRIC;
	private RunMode mode = DEF_RUN_MODE;
	
	
	
	public NearestNeighborImputation() {
		this(new NNImputationPlanner());
	}
	
	public NearestNeighborImputation(NNImputationPlanner planner) {
		super(planner);
		initFromPlanner(planner);
	}
	
	
	
	public static class NNImputationPlanner extends ImputationPlanner {
		private boolean verbose = DEF_VERBOSE;
		private int k = DEF_K;
		private Random seed = new Random();
		private double radius = DEF_RADIUS;
		private RunMode mode = DEF_RUN_MODE;
		
		public NNImputationPlanner() {}
		public NNImputationPlanner(int k) {
			this.k = k;
		}
		
		@Override
		public Random getSeed() {
			return seed;
		}

		@Override
		public boolean getVerbose() {
			return verbose;
		}
		
		public NNImputationPlanner setK(final int k) {
			this.k = k;
			return this;
		}
		
		public NNImputationPlanner setRadius(final double rad) {
			this.radius = rad;
			return this;
		}
		
		public NNImputationPlanner setRunMode(final RunMode mode) {
			this.mode = mode;
			return this;
		}
		
		@Override
		public NNImputationPlanner setSeed(final Random seed) {
			this.seed = seed;
			return this;
		}

		@Override
		public NNImputationPlanner setVerbose(boolean b) {
			this.verbose = b;
			return this;
		}
		
	}
	
	
	
	final void initFromPlanner(NNImputationPlanner planner) {
		this.k = planner.k;
		this.radius = planner.radius;
		this.mode = planner.mode;
	}
	
	
	
	

	@Override
	public double[][] process(final double[][] dat) {
		checkMat(dat);
		
		// TODO
		
		return null;
	}

	@Override
	public Algo getLoggerTag() {
		return Algo.IMPUTE;
	}
	
	@Override
	public String getName() {
		return "NN imputation";
	}
}

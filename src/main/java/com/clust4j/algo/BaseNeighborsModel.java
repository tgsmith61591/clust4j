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

import org.apache.commons.math3.linear.RealMatrix;

import com.clust4j.GlobalState;
import com.clust4j.algo.Neighborhood;
import com.clust4j.except.ModelNotFitException;
import com.clust4j.metrics.pairwise.DistanceMetric;
import com.clust4j.metrics.pairwise.GeometricallySeparable;

abstract public class BaseNeighborsModel extends AbstractClusterer {
	private static final long serialVersionUID = 1054047329248586585L;
	
	public static final NeighborsAlgorithm DEF_ALGO = NeighborsAlgorithm.AUTO;
	public static final int DEF_LEAF_SIZE = 30;
	public static final int DEF_K = 5;
	public static final double DEF_RADIUS = 5.0;
	public final static boolean DUAL_TREE_SEARCH = false;
	public final static boolean SORT = true;

	protected Integer kNeighbors = null;
	protected Double radius = null;
	protected boolean radiusMode;
	protected int leafSize, m;
	protected double[][] fit_X;
	protected NearestNeighborHeapSearch tree;
	protected NeighborsAlgorithm alg;
	
	/** Resultant neighborhood from fit method */
	protected volatile Neighborhood res;

	interface TreeBuilder extends MetricValidator {
		public NearestNeighborHeapSearch buildTree(RealMatrix data, 
				int leafSize, BaseNeighborsModel logger);
	}
	
	public static enum NeighborsAlgorithm implements TreeBuilder {
		AUTO {

			@Override
			public NearestNeighborHeapSearch buildTree(RealMatrix data,
					int leafSize, BaseNeighborsModel logger) {
				
				NeighborsAlgorithm alg = delegateAlgorithm(data);
				return alg.buildTree(data, leafSize, logger);
			}
			
			@Override
			public boolean isValidMetric(GeometricallySeparable geo) {
				throw new UnsupportedOperationException("auto has no metric validity criteria");
			}
			
		}, 
		
		KD_TREE {

			@Override
			public NearestNeighborHeapSearch buildTree(RealMatrix data,
					int leafSize, BaseNeighborsModel logger) {
				logger.alg = this;
				return new KDTree(data, leafSize, handleMetric(this, logger), logger);
			}
			
			@Override
			public boolean isValidMetric(GeometricallySeparable g) {
				return KDTree.VALID_METRICS.contains(g.getClass());
			}
		}, 
		
		BALL_TREE {

			@Override
			public NearestNeighborHeapSearch buildTree(RealMatrix data,
					int leafSize, BaseNeighborsModel logger) {
				logger.alg = this;
				return new BallTree(data, leafSize, handleMetric(this, logger), logger);
			}
			
			@Override
			public boolean isValidMetric(GeometricallySeparable g) {
				return BallTree.VALID_METRICS.contains(g.getClass());
			}
		};
		
		private static NeighborsAlgorithm delegateAlgorithm(RealMatrix arm) {
			int mn = arm.getColumnDimension() * arm.getRowDimension();
			return mn > GlobalState.ParallelismConf.MIN_ELEMENTS ?
				BALL_TREE : KD_TREE;
		}
		
		private static DistanceMetric handleMetric(NeighborsAlgorithm na, BaseNeighborsModel logger) {
			GeometricallySeparable g = logger.dist_metric;
			if(!na.isValidMetric(g)) {
				logger.warn(g.getName()+" is not a valid metric for " + na + ". "
					+ "Falling back to default Euclidean");
				logger.setSeparabilityMetric(DEF_DIST);
			}
			
			return (DistanceMetric) logger.dist_metric;
		}
	}
	
	@Override final public boolean isValidMetric(GeometricallySeparable g) {
		return this.alg.isValidMetric(g);
	}
	
	
	protected BaseNeighborsModel(AbstractClusterer caller, BaseNeighborsPlanner<? extends BaseNeighborsModel> planner) {
		super(caller, planner);
		init(planner);
	}
	
	protected BaseNeighborsModel(RealMatrix data, BaseNeighborsPlanner<? extends BaseNeighborsModel> planner, boolean as_is) {
		super(data, planner, as_is);
		init(planner);
	}
	
	protected BaseNeighborsModel(RealMatrix data, BaseNeighborsPlanner<? extends BaseNeighborsModel> planner) {
		super(data, planner);
		init(planner);
	}
	
	final private void init(BaseNeighborsPlanner<? extends BaseNeighborsModel> planner) {
		this.kNeighbors = planner.getK();
		this.radius = planner.getRadius();
		this.leafSize = planner.getLeafSize();
		
		radiusMode = null != radius;
		
		/*
		if(!(planner.getSep() instanceof DistanceMetric)) {
			warn(planner.getSep() + " not a valid metric for neighbors models. "
				+ "Falling back to default: " + DEF_DIST);
			super.setSeparabilityMetric(DEF_DIST);
		}
		*/
		
		if(leafSize < 1)
			throw new IllegalArgumentException("leafsize must be positive");
		
		/*
		 * Internally handles metric validation...
		 */
		this.tree = planner.getAlgorithm().buildTree(this.data, this.leafSize, this);
		
		// Get the data ref from the tree
		fit_X = tree.getData();
		this.m = fit_X.length;
	}

	abstract public static class BaseNeighborsPlanner<T extends BaseNeighborsModel> 
			extends BaseClustererParameters 
			implements NeighborsClassifierParameters<T> {
		private static final long serialVersionUID = 8356804193088162871L;
		
		protected int leafSize = DEF_LEAF_SIZE;
		protected NeighborsAlgorithm algo = DEF_ALGO;
		
		@Override abstract public T fitNewModel(RealMatrix d);
		abstract public BaseNeighborsPlanner<T> setAlgorithm(NeighborsAlgorithm algo);
		abstract public Integer getK();
		abstract public Double getRadius();
		
		final public int getLeafSize() { return leafSize; }
		final public NeighborsAlgorithm getAlgorithm() { return algo; }
	}
	
	public Neighborhood getNeighbors() {
		if(null == res)
			throw new ModelNotFitException("model not yet fit");
		return res.copy();
	}
	
	/**
	 * A class to query the tree for neighborhoods in parallel
	 * @author Taylor G Smith
	 */
	abstract static class ParallelNeighborhoodSearch extends ParallelChunkingTask<Neighborhood> {
		private static final long serialVersionUID = -1600812794470325448L;
		
		final BaseNeighborsModel model;
		final double[][] distances;
		final int[][] indices;
		final int lo;
		final int hi;

		public ParallelNeighborhoodSearch(double[][] X, BaseNeighborsModel model) {
			super(X); // this auto-chunks the data
			
			this.model = model;
			this.lo = 0;
			this.hi = strategy.getNumChunks(X);
			
			/*
			 * First get the length...
			 */
			int length = 0;
			for(Chunk c: this.chunks)
				length += c.size();
			
			this.distances = new double[length][];
			this.indices = new int[length][];
		}
		
		public ParallelNeighborhoodSearch(ParallelNeighborhoodSearch task, int lo, int hi) {
			super(task);
			
			this.model = task.model;
			this.lo = lo;
			this.hi = hi;
			this.distances = task.distances;
			this.indices = task.indices;
		}

		@Override
		public Neighborhood reduce(Chunk chunk) {
			Neighborhood n = query(model.tree, chunk.get());
			
			// assign to low index, since that's how we retrieved the chunk...
			final int start = chunk.start , end = start + chunk.size();
			double[][] d = n.getDistances();
			int[][] i = n.getIndices();
			
			// Set the distances and indices in place...
			for(int j = start, idx = 0; j < end; j++, idx++) {
				this.distances[j] = d[idx];
				this.indices[j] = i[idx];
			}
			
			return n;
		}

		@Override
		protected Neighborhood compute() {
			if(hi - lo <= 1) { // generally should equal one...
				return reduce(chunks.get(lo));
			} else {
				int mid = this.lo + (this.hi - this.lo) / 2;
				ParallelNeighborhoodSearch left  = newInstance(this, this.lo, mid);
				ParallelNeighborhoodSearch right = newInstance(this, mid, this.hi);
				
				left.fork();
	            right.compute();
	            left.join();

	            return new Neighborhood(distances, indices);
			}
		}
		
		abstract ParallelNeighborhoodSearch newInstance(ParallelNeighborhoodSearch p, int lo, int hi);
		abstract Neighborhood query(NearestNeighborHeapSearch tree, double[][] X);
	}
	
	
	abstract Neighborhood getNeighbors(RealMatrix matrix);
	@Override abstract protected BaseNeighborsModel fit();
}

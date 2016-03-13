package com.clust4j.algo;

import java.util.Random;

import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.GlobalState;
import com.clust4j.algo.NearestNeighborHeapSearch.Neighborhood;
import com.clust4j.algo.preprocess.FeatureNormalization;
import com.clust4j.except.ModelNotFitException;
import com.clust4j.log.LogTimer;
import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.metrics.pairwise.GeometricallySeparable;
import com.clust4j.utils.MatUtils;

public class RadiusNeighbors extends Neighbors {
	private static final long serialVersionUID = 3620377771231699918L;

	
	
	public RadiusNeighbors(AbstractRealMatrix data) {
		this(data, DEF_RADIUS);
	}
	
	public RadiusNeighbors(AbstractRealMatrix data, double radius) {
		this(data, new RadiusNeighborsPlanner(radius));
	}
	
	protected RadiusNeighbors(AbstractClusterer caller, double radius) {
		this(caller, new RadiusNeighborsPlanner(radius));
	}

	public RadiusNeighbors(AbstractRealMatrix data, RadiusNeighborsPlanner planner) {
		super(data, planner);
		validateRadius(planner.radius);
		logModelSummary();
	}
	
	protected RadiusNeighbors(AbstractClusterer caller, RadiusNeighborsPlanner planner) {
		super(caller, planner);
		validateRadius(planner.radius);
		logModelSummary();
	}
	
	static void validateRadius(double radius) {
		if(radius <= 0) throw new IllegalArgumentException("radius must be positive");
	}
	
	@Override
	final protected ModelSummary modelSummary() {
		return new ModelSummary(new Object[]{
				"Num Rows","Num Cols","Metric","Algo","Radius","Leaf Size","Scale","Force Par.","Allow Par."
			}, new Object[]{
				m,data.getColumnDimension(),getSeparabilityMetric(),
				alg, radius, leafSize, normalized,
				GlobalState.ParallelismConf.FORCE_PARALLELISM_WHERE_POSSIBLE,
				parallel
			});
	}
	
	@Override
	final protected Object[] getModelFitSummaryHeaders() {
		return new Object[]{
			"Instance","Num. Neighbors","Nrst Nbr","Avg Nbr Dist","Farthest Nbr","Wall"
		};
	}
	

	
	
	public static class RadiusNeighborsPlanner extends NeighborsPlanner {
		private static final long serialVersionUID = 2183556008789826257L;
		
		private Algorithm algo = DEF_ALGO;
		private GeometricallySeparable dist= NearestNeighborHeapSearch.DEF_DIST;
		private FeatureNormalization norm = DEF_NORMALIZER;
		private boolean verbose = DEF_VERBOSE;
		private boolean scale = DEF_SCALE;
		private Random seed = DEF_SEED;
		private double radius;
		private int leafSize = DEF_LEAF_SIZE;
		
		
		public RadiusNeighborsPlanner() { this(DEF_RADIUS); }
		public RadiusNeighborsPlanner(double rad) {
			this.radius = rad;
		}
		

		
		@Override
		public RadiusNeighbors buildNewModelInstance(AbstractRealMatrix data) {
			return new RadiusNeighbors(data, this.copy());
		}

		@Override
		public RadiusNeighborsPlanner setAlgorithm(Algorithm algo) {
			this.algo = algo;
			return this;
		}

		@Override
		public Algorithm getAlgorithm() {
			return algo;
		}

		@Override
		public RadiusNeighborsPlanner copy() {
			return new RadiusNeighborsPlanner(radius)
				.setAlgorithm(algo)
				.setNormalizer(norm)
				.setScale(scale)
				.setSeed(seed)
				.setSep(dist)
				.setVerbose(verbose)
				.setLeafSize(leafSize);
		}
		
		@Override
		public int getLeafSize() {
			return leafSize;
		}
		
		@Override
		final public Integer getK() {
			return null;
		}

		@Override
		final public Double getRadius() {
			return radius;
		}
		
		@Override
		public FeatureNormalization getNormalizer() {
			return norm;
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

		public RadiusNeighborsPlanner setLeafSize(int leafSize) {
			this.leafSize = leafSize;
			return this;
		}
		
		@Override
		public RadiusNeighborsPlanner setNormalizer(FeatureNormalization norm) {
			this.norm = norm;
			return this;
		}

		@Override
		public RadiusNeighborsPlanner setScale(boolean b) {
			this.scale = b;
			return this;
		}

		@Override
		public RadiusNeighborsPlanner setSeed(Random rand) {
			this.seed= rand;
			return this;
		}

		@Override
		public RadiusNeighborsPlanner setVerbose(boolean b) {
			this.verbose = b;
			return this;
		}

		@Override
		public RadiusNeighborsPlanner setSep(GeometricallySeparable dist) {
			this.dist = dist;
			return this;
		}
	}


	@Override
	public String getName() {
		return "RadiusNeighbors";
	}
	
	public double getRadius() {
		return radius;
	}
	
	@Override
	public boolean equals(Object o) {
		if(this == o)
			return true;
		if(o instanceof RadiusNeighbors) {
			RadiusNeighbors other = (RadiusNeighbors)o;
			
			
			return 
				((null == other.radius || null == this.radius) ?
					other.radius == this.radius : 
						other.radius.intValue() == this.radius)
				&& other.leafSize == this.leafSize
				&& MatUtils.equalsExactly(other.fit_X, this.fit_X);
		}
		
		return false;
	}

	@Override
	public RadiusNeighbors fit() {
		synchronized(this) {
			try {
				if(null != res)
					return this;

				final LogTimer timer = new LogTimer();
				Neighborhood initRes = new Neighborhood(tree.queryRadius(fit_X, radius, false));
				info("queried "+this.alg+" for radius neighbors in " + timer.toString());
				
				
				double[][] dists = initRes.getDistances();
				int[][] indices  = initRes.getIndices();
				int[] tmp_ind_neigh, ind_neighbor;
				double[] tmp_dists, dist_row;
				
				
				for(int ind = 0; ind < indices.length; ind++) {
					ind_neighbor = indices[ind];
					dist_row = dists[ind];
					
					// Keep track for summary
					double v, sum = 0,
						minDist = Double.POSITIVE_INFINITY, 
						maxDist = Double.NEGATIVE_INFINITY;
					
					int b_count = 0;
					boolean b_val;
					boolean[] mask = new boolean[ind_neighbor.length];
					for(int j = 0; j < ind_neighbor.length; j++) {
						b_val = ind_neighbor[j] != ind;
						mask[j] = b_val;
						v = dist_row[j];
						
						if(b_val) {
							sum += v;
							minDist = FastMath.min(minDist, v);
							maxDist = FastMath.max(maxDist, v);
							b_count++;
						}
					}
					
					tmp_ind_neigh = new int[b_count];
					tmp_dists = new double[b_count];
					
					for(int j = 0, k = 0; j < mask.length; j++) {
						if(mask[j]) {
							tmp_ind_neigh[k] = ind_neighbor[j];
							tmp_dists[k] = dist_row[j];
							k++;
						}
					}
					
					indices[ind] = tmp_ind_neigh;
					dists[ind] = tmp_dists;
					
					fitSummary.add(new Object[]{ind, b_count, minDist, (double)sum/(double)b_count, maxDist, timer.wallTime()});
				}
				
				res = new Neighborhood(dists, indices);
				
				sayBye(timer);
				return this;
			} catch(OutOfMemoryError | StackOverflowError e) {
				error(e.getLocalizedMessage() + " - ran out of memory during model fitting");
				throw e;
			} // end try/catch
			
		} // End synch
	}

	@Override
	public Neighborhood getNeighbors(AbstractRealMatrix x) {
		return getNeighbors(x, radius);
	}
	
	public Neighborhood getNeighbors(AbstractRealMatrix x, double rad) {
		return getNeighbors(x.getData(), rad);
	}
	
	Neighborhood getNeighbors(double[][] X, double rad) {
		if(null == res)
			throw new ModelNotFitException("model not yet fit");
		
		validateRadius(rad);
		return new Neighborhood(tree.queryRadius(X, rad, false));
	}

	@Override
	public Algo getLoggerTag() {
		return Algo.RADIUS;
	}
}

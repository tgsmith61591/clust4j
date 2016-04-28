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

import java.util.concurrent.RejectedExecutionException;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.algo.Neighborhood;
import com.clust4j.except.ModelNotFitException;
import com.clust4j.log.LogTimer;
import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.utils.MatUtils;

final public class RadiusNeighbors extends BaseNeighborsModel {
	private static final long serialVersionUID = 3620377771231699918L;
	
	
	protected RadiusNeighbors(RealMatrix data) {
		this(data, DEF_RADIUS);
	}
	
	protected RadiusNeighbors(RealMatrix data, double radius) {
		this(data, new RadiusNeighborsParameters(radius));
	}
	
	protected RadiusNeighbors(AbstractClusterer caller, double radius) {
		this(caller, new RadiusNeighborsParameters(radius));
	}

	protected RadiusNeighbors(RealMatrix data, RadiusNeighborsParameters planner) {
		super(data, planner);
		validateRadius(planner.getRadius());
		logModelSummary();
	}
	
	protected RadiusNeighbors(AbstractClusterer caller, RadiusNeighborsParameters planner) {
		super(caller, planner);
		validateRadius(planner.getRadius());
		logModelSummary();
	}
	
	protected RadiusNeighbors(RealMatrix data, RadiusNeighborsParameters planner, boolean as_is) {
		super(data, planner, as_is);
		validateRadius(planner.getRadius());
		logModelSummary();
	}
	
	
	
	
	static void validateRadius(double radius) {
		if(radius <= 0) throw new IllegalArgumentException("radius must be positive");
	}
	
	@Override
	final protected ModelSummary modelSummary() {
		return new ModelSummary(new Object[]{
				"Num Rows","Num Cols","Metric","Algo","Radius","Leaf Size","Allow Par."
			}, new Object[]{
				m,data.getColumnDimension(),getSeparabilityMetric(),
				alg, radius, leafSize,
				parallel
			});
	}
	
	@Override
	final protected Object[] getModelFitSummaryHeaders() {
		return new Object[]{
			"Instance","Num. Neighbors","Nrst Nbr","Avg Nbr Dist","Farthest Nbr","Wall"
		};
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
			
			
			return super.equals(o)
				&& ((null == other.radius || null == this.radius) ?
					other.radius == this.radius : 
						other.radius.intValue() == this.radius)
				&& other.leafSize == this.leafSize
				&& MatUtils.equalsExactly(other.fit_X, this.fit_X);
		}
		
		return false;
	}

	@Override
	protected RadiusNeighbors fit() {
		synchronized(fitLock) {
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
		}
	}

	@Override
	public Neighborhood getNeighbors(RealMatrix x) {
		return getNeighbors(x, radius);
	}
	
	/**
	 * For internal use
	 * @param x
	 * @param parallelize
	 * @return
	 */
	protected Neighborhood getNeighbors(double[][] x, boolean parallelize) {
		return getNeighbors(x, radius, parallelize);
	}
	
	/**
	 * For internal use
	 * @param x
	 * @return
	 */
	protected Neighborhood getNeighbors(double[][] x) {
		return getNeighbors(x, radius, false);
	}
	
	public Neighborhood getNeighbors(RealMatrix x, double rad) {
		return getNeighbors(x.getData(), rad, parallel);
	}
	
	protected Neighborhood getNeighbors(double[][] X, double rad, boolean parallelize) {
		if(null == res)
			throw new ModelNotFitException("model not yet fit");
		validateRadius(rad);
		
		/*
		 * Try parallel if we can...
		 */
		if(parallelize) {
			try {
				return ParallelRadSearch.doAll(X, this, rad);
			} catch(RejectedExecutionException r) {
				warn("parallel neighborhood search failed; falling back to serial search");
			}
		}
		
		return tree.queryRadius(X, rad, false);
	}
	
	
	/**
	 * A class to query the tree for neighborhoods in parallel
	 * @author Taylor G Smith
	 */
	static class ParallelRadSearch extends ParallelNeighborhoodSearch {
		private static final long serialVersionUID = -1600812794470325448L;
		final double rad;

		public ParallelRadSearch(double[][] X, RadiusNeighbors model, final double rad) {
			super(X, model); // this auto-chunks the data
			this.rad = rad;
		}
		
		public ParallelRadSearch(ParallelRadSearch task, int lo, int hi) {
			super(task, lo, hi);
			this.rad = task.rad;
		}
		
		static Neighborhood doAll(double[][] X, RadiusNeighbors nn, double rad) {
			return getThreadPool().invoke(new ParallelRadSearch(X, nn, rad));
		}

		@Override
		ParallelRadSearch newInstance(ParallelNeighborhoodSearch p, int lo, int hi) {
			return new ParallelRadSearch((ParallelRadSearch)p, lo, hi);
		}

		@Override
		Neighborhood query(NearestNeighborHeapSearch tree, double[][] X) {
			return tree.queryRadius(X, rad, false);
		}
	}
	

	@Override
	public Algo getLoggerTag() {
		return Algo.RADIUS;
	}
}

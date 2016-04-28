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
import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.log.LogTimer;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;

final public class NearestNeighbors extends BaseNeighborsModel {
	private static final long serialVersionUID = 8306843374522289973L;

	
	protected NearestNeighbors(RealMatrix data) {
		this(data, DEF_K);
	}
	
	protected NearestNeighbors(AbstractClusterer caller) {
		this(caller, DEF_K);
	}

	protected NearestNeighbors(RealMatrix data, int k) {
		this(data, new NearestNeighborsParameters(k));
	}
	
	protected NearestNeighbors(AbstractClusterer caller, int k) {
		this(caller, new NearestNeighborsParameters(k));
	}

	protected NearestNeighbors(RealMatrix data, NearestNeighborsParameters planner) {
		this(data, planner, false);
	}
	
	protected NearestNeighbors(AbstractClusterer caller, NearestNeighborsParameters planner) {
		super(caller, planner);
		validateK(kNeighbors, m);
		logModelSummary();
	}
	
	protected NearestNeighbors(RealMatrix data, NearestNeighborsParameters planner, boolean as_is) {
		super(data, planner, as_is);
		validateK(kNeighbors, m);
		logModelSummary();
	}
	
	
	
	
	private static void validateK(int k, int m) {
		if(k < 1) throw new IllegalArgumentException("k must be positive");
		if(k > m) throw new IllegalArgumentException("k must be <= number of samples");
	}
	
	@Override
	final protected ModelSummary modelSummary() {
		return new ModelSummary(new Object[]{
				"Num Rows","Num Cols","Metric","Algo","K","Leaf Size","Allow Par."
			}, new Object[]{
				m,data.getColumnDimension(),getSeparabilityMetric(),
				alg, kNeighbors, leafSize,
				parallel
			});
	}
	
	
	
	
	@Override
	public boolean equals(Object o) {
		if(this == o)
			return true;
		if(o instanceof NearestNeighbors) {
			NearestNeighbors other = (NearestNeighbors)o;
			
			return super.equals(o) // UUID check
				&& ((null == other.kNeighbors || null == this.kNeighbors) ?
					other.kNeighbors == this.kNeighbors : 
						other.kNeighbors.intValue() == this.kNeighbors)
				&& other.leafSize == this.leafSize
				&& MatUtils.equalsExactly(other.fit_X, this.fit_X);
		}
		
		return false;
	}
	
	@Override
	public String getName() {
		return "NearestNeighbors";
	}
	
	public int getK() {
		return kNeighbors;
	}

	@Override
	protected NearestNeighbors fit() {
		synchronized(fitLock) {
		
			if(null != res)
				return this;
			
			
			// CORNER! If k == m, we can't do kNeighbors + 1..
			int nNeighbors = FastMath.min(kNeighbors + 1, m); //kNeighbors + 1;
			final LogTimer timer = new LogTimer();
			
			// We can do parallel here!
			Neighborhood initRes = null;
			if(parallel) {
				try {
					initRes = ParallelNNSearch.doAll(fit_X, this, nNeighbors);
				} catch(RejectedExecutionException r) {
					warn("parallel neighborhood search failed; falling back to serial query");
				}
			}
			
			// Gets here in serial mode or if parallel failed...
			if(null == initRes)
				initRes = new Neighborhood(tree.query(fit_X, nNeighbors, DUAL_TREE_SEARCH, SORT));
			info("queried "+this.alg+" for nearest neighbors in " + timer.toString());

			
			double[][] dists = initRes.getDistances();
			int[][] indices  = initRes.getIndices();
			int i, j, ni = indices[0].length;
			
			
			// Set up sample range
			int[] sampleRange = VecUtils.arange(m);
			
			
			boolean allInRow, bval;
			boolean[] dupGroups = new boolean[m];
			boolean[][] sampleMask= new boolean[m][ni];
			for(i = 0; i < m; i++) {
				allInRow = true;
				
				for(j = 0; j < ni; j++) {
					bval = indices[i][j] != sampleRange[i];
					sampleMask[i][j] = bval;
					allInRow &= bval;
				}
				
				dupGroups[i] = allInRow; // duplicates in row?
			}
			
			
			// Comment from SKLEARN:
			// Corner case: When the number of duplicates are more
	        // than the number of neighbors, the first NN will not
	        // be the sample, but a duplicate.
	        // In that case mask the first duplicate.
			// sample_mask[:, 0][dup_gr_nbrs] = False
			
			for(i = 0; i < m; i++)
				if(dupGroups[i])
					sampleMask[i][0] = false;
			
			
			// Build output indices
			int k = 0;
			int[] indOut = new int[m * (nNeighbors - 1)];
			double[] distOut = new double[m * (nNeighbors - 1)];
			for(i = 0; i < m; i++) {
				double minDist = Double.POSITIVE_INFINITY, maxDist = Double.NEGATIVE_INFINITY;
				
				for(j = 0; j < ni; j++) {
					if(sampleMask[i][j]) {
						indOut[k] = indices[i][j];
						distOut[k]= dists[i][j];
						
						minDist = FastMath.min(dists[i][j], minDist);
						maxDist = FastMath.max(dists[i][j], maxDist);
						
						k++;
					}
				}
				
				fitSummary.add(new Object[]{
					i, minDist, maxDist, timer.wallTime()
				});
			}
			
			res = new Neighborhood(
				MatUtils.reshape(distOut, m, nNeighbors - 1),
				MatUtils.reshape(indOut,  m, nNeighbors - 1));
			
			
			sayBye(timer);
			return this;
		}

	}
	
	@Override
	final protected Object[] getModelFitSummaryHeaders() {
		return new Object[]{
			"Instance","Nrst-Nbr. Dist","Max-Nbr. Dist","Wall"
		};
	}
	
	@Override
	public Neighborhood getNeighbors(RealMatrix x) {
		return getNeighbors(x, kNeighbors);
	}
	
	/**
	 * For internal use
	 * @param x
	 * @param parallelize
	 * @return
	 */
	protected Neighborhood getNeighbors(double[][] x, boolean parallelize) {
		return getNeighbors(x, kNeighbors, parallelize);
	}
	
	/**
	 * For internal use
	 * @param x
	 * @return
	 */
	protected Neighborhood getNeighbors(double[][] x) {
		return getNeighbors(x, kNeighbors, false);
	}
	
	public Neighborhood getNeighbors(RealMatrix x, int k) {
		return getNeighbors(x.getData(), k, parallel);
	}
	
	/**
	 * For internal use
	 * @param X
	 * @param k
	 * @return
	 */
	protected Neighborhood getNeighbors(double[][] X, int k, boolean parallelize) {
		if(null == res)
			throw new ModelNotFitException("model not yet fit");
		
		validateK(k, m); // Should be X.length  or m??
		
		/*
		 * Try parallel if we can...
		 */
		if(parallelize) {
			try {
				return ParallelNNSearch.doAll(X, this, k);
			} catch(RejectedExecutionException r) {
				warn("parallel neighborhood search failed; falling back to serial search");
			}
		}
		
		return tree.query(X, k, DUAL_TREE_SEARCH, SORT);
	}
	
	/**
	 * A class to query the tree for neighborhoods in parallel
	 * @author Taylor G Smith
	 */
	static class ParallelNNSearch extends ParallelNeighborhoodSearch {
		private static final long serialVersionUID = -1600812794470325448L;
		final int k;

		public ParallelNNSearch(double[][] X, NearestNeighbors model, final int k) {
			super(X, model); // this auto-chunks the data
			this.k = k;
		}
		
		public ParallelNNSearch(ParallelNNSearch task, int lo, int hi) {
			super(task, lo, hi);
			this.k = task.k;
		}
		
		static Neighborhood doAll(double[][] X, NearestNeighbors nn, int k) {
			return getThreadPool().invoke(new ParallelNNSearch(X, nn, k));
		}

		@Override
		ParallelNNSearch newInstance(ParallelNeighborhoodSearch p, int lo, int hi) {
			return new ParallelNNSearch((ParallelNNSearch)p, lo, hi);
		}

		@Override
		Neighborhood query(NearestNeighborHeapSearch tree, double[][] X) {
			return tree.query(X, k, DUAL_TREE_SEARCH, SORT);
		}
	}
	
	

	@Override
	public Algo getLoggerTag() {
		return Algo.NEAREST;
	}
}

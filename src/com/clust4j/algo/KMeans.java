package com.clust4j.algo;

import java.util.ArrayList;
import java.util.Random;

import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.algo.NearestCentroid.NearestCentroidPlanner;
import com.clust4j.algo.preprocess.FeatureNormalization;
import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.log.LogTimer;
import com.clust4j.metrics.pairwise.Distance;
import com.clust4j.metrics.pairwise.GeometricallySeparable;
import com.clust4j.utils.EntryPair;
import com.clust4j.utils.VecUtils;

/**
 * <a href="https://en.wikipedia.org/wiki/K-means_clustering">KMeans clustering</a> is
 * a method of vector quantization, originally from signal processing, that is popular 
 * for cluster analysis in data mining. KMeans clustering aims to partition <i>m</i> 
 * observations into <i>k</i> clusters in which each observation belongs to the cluster 
 * with the nearest mean, serving as a prototype of the cluster. This results in 
 * a partitioning of the data space into <a href="https://en.wikipedia.org/wiki/Voronoi_cell">Voronoi cells</a>.
 * 
 * @author Taylor G Smith &lt;tgsmith61591@gmail.com&gt;
 */
public class KMeans extends AbstractCentroidClusterer {
	private static final long serialVersionUID = 1102324012006818767L;
	final public static GeometricallySeparable DEF_DIST = Distance.EUCLIDEAN;
	final public static int DEF_MAX_ITER = 100;
	
	
	public KMeans(final AbstractRealMatrix data) {
		this(data, DEF_K);
	}
	
	public KMeans(final AbstractRealMatrix data, final int k) {
		this(data, new KMeansPlanner(k));
	}
	
	public KMeans(final AbstractRealMatrix data, final KMeansPlanner planner) {
		super(data, planner);
	}
	
	
	public static class KMeansPlanner extends CentroidClustererPlanner {
		private static final long serialVersionUID = -813106538623499760L;
		
		private InitializationStrategy strat = DEF_INIT;
		private FeatureNormalization norm = DEF_NORMALIZER;
		private int maxIter = DEF_MAX_ITER;
		private double minChange = DEF_CONVERGENCE_TOLERANCE;
		private GeometricallySeparable dist = DEF_DIST;
		private boolean verbose = DEF_VERBOSE;
		private boolean scale = DEF_SCALE;
		private Random seed = DEF_SEED;
		private int k = DEF_K;
		
		public KMeansPlanner() { }
		public KMeansPlanner(int k) {
			this.k = k;
		}
		
		@Override
		public KMeans buildNewModelInstance(final AbstractRealMatrix data) {
			return new KMeans(data, this.copy());
		}
		
		@Override
		public KMeansPlanner copy() {
			return new KMeansPlanner(k)
				.setMaxIter(maxIter)
				.setConvergenceCriteria(minChange)
				.setScale(scale)
				.setSep(dist)
				.setVerbose(verbose)
				.setSeed(seed)
				.setNormalizer(norm)
				.setInitializationStrategy(strat);
		}
		
		@Override
		public int getK() {
			return k;
		}
		
		@Override
		public InitializationStrategy getInitializationStrategy() {
			return strat;
		}
		
		@Override
		public int getMaxIter() {
			return maxIter;
		}
		
		@Override
		public double getConvergenceTolerance() {
			return minChange;
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
		public GeometricallySeparable getSep() {
			return dist;
		}
		
		@Override
		public boolean getVerbose() {
			return verbose;
		}
		
		@Override
		public KMeansPlanner setSep(final GeometricallySeparable dist) {
			this.dist = dist;
			return this;
		}
		
		public KMeansPlanner setMaxIter(final int max) {
			this.maxIter = max;
			return this;
		}

		@Override
		public KMeansPlanner setConvergenceCriteria(final double min) {
			this.minChange = min;
			return this;
		}
		
		@Override
		public KMeansPlanner setInitializationStrategy(InitializationStrategy init) {
			this.strat = init;
			return this;
		}
		
		@Override
		public KMeansPlanner setScale(final boolean scale) {
			this.scale = scale;
			return this;
		}
		
		@Override
		public KMeansPlanner setSeed(final Random seed) {
			this.seed = seed;
			return this;
		}
		
		@Override
		public KMeansPlanner setVerbose(final boolean v) {
			this.verbose = v;
			return this;
		}

		@Override
		public FeatureNormalization getNormalizer() {
			return norm;
		}

		@Override
		public KMeansPlanner setNormalizer(FeatureNormalization norm) {
			this.norm = norm;
			return this;
		}
	}
	
	
	@Override
	public String getName() {
		return "KMeans";
	}
	
	@Override
	final public KMeans fit() {
		synchronized(this) {
			
			try {
				if(null != labels) // already fit
					return this;
				

				final LogTimer timer = new LogTimer();
				final double[][] X = data.getData();
				final int n = data.getColumnDimension();
				
				
				// Corner case: K = 1
				if(1 == k) {
					labelFromSingularK(X);
					fitSummary.add(new Object[]{ iter, converged, cost, cost, timer.wallTime() });
					sayBye(timer);
					return this;
				}
				
				
				
				// Nearest centroid model to predict labels
				NearestCentroid model;
				EntryPair<int[], double[]> label_dist;
				
				
				// Keep track of TSS (sum of barycentric distances)
				double maxCost = Double.NEGATIVE_INFINITY;
				cost = Double.POSITIVE_INFINITY;
				ArrayList<double[]> new_centroids;
				
				
				for(iter = 0; iter < maxIter; iter++) {
					
					// Get labels for nearest centroids
					model = new NearestCentroid(centroidsToMatrix(), 
						VecUtils.arange(k), new NearestCentroidPlanner()
							.setScale(false) // already scaled maybe
							.setSeed(getSeed())
							.setSep(getSeparabilityMetric())
							.setVerbose(false)).fit();
					label_dist = model.predict(X);
					
					// unpack the EntryPair
					labels = label_dist.getKey();
					
					
					// Start by computing TSS using barycentric dist
					double system_cost = 0.0;
					double[] centroid, new_centroid;
					new_centroids = new ArrayList<>(k);
					for(int i = 0; i < k; i++) {
						centroid = centroids.get(i);
						new_centroid = new double[n];
						
						// Compute the current cost for each cluster,
						// break if difference in TSS < tol. Otherwise
						// update the centroids to means of clusters.
						// We can compute what the new clusters will be
						// here, but don't assign yet
						int label, count = 0;
						double clust_cost = 0;
						for(int row = 0; row < m; row++) {
							label = labels[row];
							double diff;
							
							if(label == i) {
								for(int j = 0; j < n; j++) {
									new_centroid[j] += X[row][j];
									diff = X[row][j] - centroid[j];
									clust_cost += diff * diff;
								}
								
								// number in cluster
								count++;
							}
						}
						
						// Update the new centroid (currently a sum) to be a mean
						for(int j = 0; j < n; j++)
							new_centroid[j] /= (double)count;
						new_centroids.add(new_centroid);
						
						// Update system cost
						system_cost += clust_cost;
						
					} // end centroid re-assignment
					
					
					
					// Add current state to fitSummary
					fitSummary.add(new Object[]{ iter, converged, maxCost, cost, timer.wallTime() });
					
					
					// Assign new centroids
					centroids = new_centroids;
					double diff = cost - system_cost;	// results in Inf on first iteration
					cost = system_cost;
					
					// if diff is Inf, this is the first pass. Max is always first pass
					if(Double.isInfinite(diff))
						maxCost = cost; // should always stay the same..
					
					
					
					
					// Check for convergence
					if( FastMath.abs(diff) < tolerance ) {	// Inf always returns false in comparison
						// Did converge
						converged = true;
						iter++; // Going to break and miss this..
						break;
					}
					
				} // end iterations
				

				// last one...
				fitSummary.add(new Object[]{ iter, 
					converged, maxCost, cost, timer.wallTime() });
				
				if(!converged)
					warn("algorithm did not converge");
					
				
				// wrap things up, create summary..
				reorderLabelsAndCentroids();
				sayBye(timer);
				
				
				return this;
				
			} catch(OutOfMemoryError | StackOverflowError e) {
				error(e.getLocalizedMessage() + " - ran out of memory during model fitting");
				throw e;
			} // end try/catch
			
		} // end sync
	}
	

	@Override
	public Algo getLoggerTag() {
		return com.clust4j.log.Log.Tag.Algo.KMEANS;
	}

	@Override
	protected Object[] getModelFitSummaryHeaders() {
		return new Object[]{
			"Iter. #","Converged","Max Cost","Min Cost","Wall"
		};
	}
}

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
import java.util.TreeMap;

import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.algo.NearestCentroidParameters;
import com.clust4j.except.NaNException;
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
final public class KMeans extends AbstractCentroidClusterer {
	private static final long serialVersionUID = 1102324012006818767L;
	final public static GeometricallySeparable DEF_DIST = Distance.EUCLIDEAN;
	final public static int DEF_MAX_ITER = 100;
	
	
	
	protected KMeans(final RealMatrix data) {
		this(data, DEF_K);
	}
	
	protected KMeans(final RealMatrix data, final int k) {
		this(data, new KMeansParameters(k));
	}
	
	protected KMeans(final RealMatrix data, final KMeansParameters planner) {
		super(data, planner);
	}
	
	
	
	
	@Override
	public String getName() {
		return "KMeans";
	}
	
	@Override
	protected KMeans fit() {
		synchronized(fitLock) {

			if(null != labels) // already fit
				return this;
			

			final LogTimer timer = new LogTimer();
			final double[][] X = data.getData();
			final int n = data.getColumnDimension();
			final double nan = Double.NaN;
			
			
			// Corner case: K = 1 or all singular values
			if(1 == k) {
				labelFromSingularK(X);
				fitSummary.add(new Object[]{ iter, converged, tss, tss, nan, timer.wallTime() });
				sayBye(timer);
				return this;
			}
			
			
			
			// Nearest centroid model to predict labels
			NearestCentroid model = null;
			EntryPair<int[], double[]> label_dist;
			
			
			// Keep track of TSS (sum of barycentric distances)
			double last_wss_sum = Double.POSITIVE_INFINITY, wss_sum = 0;
			ArrayList<double[]> new_centroids;
			
			for(iter = 0; iter < maxIter; iter++) {
				
				// Get labels for nearest centroids
				try {
					model = new NearestCentroid(CentroidUtils.centroidsToMatrix(centroids, false), 
						VecUtils.arange(k), new NearestCentroidParameters()
							.setSeed(getSeed())
							.setMetric(getSeparabilityMetric())
							.setVerbose(false)).fit();
				} catch(NaNException NaN) {
					/*
					 * If they metric used produces lots of infs or -infs, it 
					 * makes it hard if not impossible to effectively segment the
					 * input space. Thus, the centroid assignment portion below can
					 * yield a zero count (denominator) for one or more of the centroids
					 * which makes the entire row NaN. We should tell the user to
					 * try a different metric, if that's the case.
					 *
					error(new IllegalClusterStateException(dist_metric.getName()+" produced an entirely " +
					  "infinite distance matrix, making it difficult to segment the input space. Try a different " +
					  "metric."));
					 */
					this.k = 1;
					warn("(dis)similarity metric ("+dist_metric+") cannot partition space without propagating Infs. Returning one cluster");
					
					labelFromSingularK(X);
					fitSummary.add(new Object[]{ iter, converged, tss, tss, nan, timer.wallTime() });
					sayBye(timer);
					return this;
				}
				
				label_dist = model.predict(X);
				
				// unpack the EntryPair
				labels = label_dist.getKey();
				new_centroids = new ArrayList<>(k);
				
				
				int label;
				wss = new double[k];
				int[] centroid_counts = new int[k];
				double[] centroid;
				double[][] new_centroid_arrays = new double[k][n];
				for(int i = 0; i < m; i++) {
					label = labels[i];
					centroid = centroids.get(label);
					
					// increment count for this centroid
					double this_cost = 0;
					centroid_counts[label]++;
					for(int j = 0; j < centroid.length; j++) {
						double diff = X[i][j] - centroid[j];
						this_cost += (diff * diff);
						
						// Add the the centroid sums
						new_centroid_arrays[label][j] += X[i][j];
					}
					
					// add this cost to the WSS
					wss[label] += this_cost;
				}
				
				// one pass of K for some consolidation
				wss_sum = 0;
				for(int i = 0; i < k; i++) {
					wss_sum += wss[i];
					
					for(int j = 0; j < n; j++) // meanify
						new_centroid_arrays[i][j] /= (double)centroid_counts[i];
					
					new_centroids.add(new_centroid_arrays[i]);
				}
				
				// update the BSS
				bss = tss - wss_sum;
				
				
				
				// Assign new centroids
				double diff = last_wss_sum - wss_sum;
				last_wss_sum = wss_sum;
				
				
				// Check for convergence and add summary:
				converged = FastMath.abs(diff) < tolerance; // first iter will be inf
				fitSummary.add(new Object[]{ 
					converged ? iter++ : iter, 
					converged, 
					tss, wss_sum, bss, 
					timer.wallTime() });
				
				if(converged) {
					break;
				} else {
					// otherwise, reassign centroids
					centroids = new_centroids;
				}
				
			} // end iterations
			
			
			
			// Reorder the labels, centroids and wss indices
			reorderLabelsAndCentroids();
			
			if(!converged)
				warn("algorithm did not converge");
				
			
			// wrap things up, create summary..
			sayBye(timer);
			
			
			return this;
		}
			
	}
	

	@Override
	public Algo getLoggerTag() {
		return com.clust4j.log.Log.Tag.Algo.KMEANS;
	}

	@Override
	protected Object[] getModelFitSummaryHeaders() {
		return new Object[]{
			"Iter. #","Converged","TSS","WSS","BSS","Wall"
		};
	}
	
	/**
	 * Reorder the labels in order of appearance using the 
	 * {@link LabelEncoder}. Also reorder the centroids to correspond
	 * with new label order
	 */
	@Override
	protected void reorderLabelsAndCentroids() {
		boolean wss_null = null == wss;
		
		/*
		 *  reorder labels...
		 */
		final LabelEncoder encoder = new LabelEncoder(labels).fit();
		labels = encoder.getEncodedLabels();
		
		// also reorder centroids... takes O(2K) passes
		TreeMap<Integer, double[]> tmpCentroids = new TreeMap<>();
		double[] new_wss = new double[k];
		
		/*
		 * We have to be delicate about this--KMedoids stores
		 * labels as indices pointing to which record is the medoid,
		 * whereas KMeans uses 0 thru K. Thus we can simply index in
		 * KMeans, but will get an IndexOOB exception in Kmedoids, so
		 * we need to come up with a universal solution which might
		 * look ugly at a glance, but is robust to both.
		 */
		int encoded;
		for(int i = 0; i < k; i++) {
			encoded = encoder.reverseEncodeOrNull(i);
			tmpCentroids.put(i, centroids.get(encoded));
			
			new_wss[i] = wss_null ? Double.NaN : wss[encoded];
		}
		
		for(int i = 0; i < k; i++)
			centroids.set(i, tmpCentroids.get(i));
		
		// reset wss
		this.wss = new_wss;
	}
}

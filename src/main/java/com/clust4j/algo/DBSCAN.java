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
import java.util.HashSet;
import java.util.Stack;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.RealMatrix;

import com.clust4j.algo.RadiusNeighborsParameters;
import com.clust4j.log.LogTimer;
import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.metrics.pairwise.GeometricallySeparable;
import com.clust4j.metrics.pairwise.SimilarityMetric;
import com.clust4j.utils.MatUtils;


/**
 * <a href="https://en.wikipedia.org/wiki/DBSCAN">DBSCAN</a> (Density Based Spatial Clustering
 * for Applications with Noise) is a data clustering algorithm proposed by Martin Ester, 
 * Hans-Peter Kriegel, Jorg Sander and Xiaowei Xu in 1996. It is a density-based clustering 
 * algorithm: given a set of points in some space, it groups together points that are 
 * closely packed together (points with many nearby neighbors), marking as outliers 
 * points that lie alone in low-density regions (whose nearest neighbors are too far away).
 * 
 * @see <a href="http://www.dbs.ifi.lmu.de/Publikationen/Papers/KDD-96.final.frame.pdf">DBSCAN, 
 * A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise</a>
 * @see {@link AbstractDensityClusterer}
 * @author Taylor G Smith &lt;tgsmith61591@gmail.com&gt;, adapted from sklearn implementation by Lars Buitinck
 *
 */
final public class DBSCAN extends AbstractDBSCAN {
	/**
	 * 
	 */
	private static final long serialVersionUID = 6749407933012974992L;
	final private int m;
	final public static HashSet<Class<? extends GeometricallySeparable>> UNSUPPORTED_METRICS;
	
	
	/**
	 * Static initializer
	 */
	static {
		UNSUPPORTED_METRICS = new HashSet<>();
		// Add metrics here if necessary...
	}
	
	@Override final public boolean isValidMetric(GeometricallySeparable geo) {
		return !UNSUPPORTED_METRICS.contains(geo.getClass()) && !(geo instanceof SimilarityMetric);
	}
	
	// Race conditions exist in retrieving either one of these...
	private volatile int[] labels = null;
	private volatile double[] sampleWeights = null;
	private volatile boolean[] coreSamples = null;
	private volatile int numClusters;
	private volatile int numNoisey;
	
	
	
	/**
	 * Constructs an instance of DBSCAN from the default epsilon
	 * @param data
	 */
	protected DBSCAN(final RealMatrix data) {
		this(data, DEF_EPS);
	}
	
	
	/**
	 * Constructs an instance of DBSCAN from the default planner values
	 * @param eps
	 * @param data
	 */
	protected DBSCAN(final RealMatrix data, final double eps) {
		this(data, new DBSCANParameters(eps));
	}
	
	/**
	 * Constructs an instance of DBSCAN from the provided builder
	 * @param builder
	 * @param data
	 */
	protected DBSCAN(final RealMatrix data, final DBSCANParameters planner) {
		super(data, planner);
		this.m = data.getRowDimension();
		this.eps = planner.getEps();
		
		// Error handle...
		if(this.eps <= 0.0) 
			error(new IllegalArgumentException("eps "
				+ "must be greater than 0.0"));
		
		if(!isValidMetric(this.dist_metric)) {
			warn(this.dist_metric.getName() + " is not valid for "+getName()+". "
				+ "Falling back to default Euclidean dist");
			setSeparabilityMetric(DEF_DIST);
		}
		
		logModelSummary();
	}
	
	@Override
	final protected ModelSummary modelSummary() {
		return new ModelSummary(new Object[]{
				"Num Rows","Num Cols","Metric","Epsilon","Min Pts.","Allow Par."
			}, new Object[]{
				m,data.getColumnDimension(),getSeparabilityMetric(),
				eps, minPts,
				parallel
			});
	}
	

	
	
	@Override
	public boolean equals(Object o) {
		if(this == o)
			return true;
		if(o instanceof DBSCAN) {
			DBSCAN d = (DBSCAN)o;
			
			/*
			 * This is a litmus test of
			 * whether the model has been fit yet.
			 */
			if(null == this.labels ^ null == d.labels)
				return false;
			
			return super.equals(o) // tests for UUID
				&& MatUtils.equalsExactly(this.data.getDataRef(), d.data.getDataRef())
				&& this.eps == d.eps;
		}
		
		return false;
	}
	
	public double getEps() {
		return eps;
	}
	
	@Override
	public int[] getLabels() {
		return super.handleLabelCopy(labels);
	}
	
	@Override
	public String getName() {
		return "DBSCAN";
	}
	
	@Override
	protected DBSCAN fit() {
		synchronized(fitLock) {

			if(null!=labels) // Then we've already fit this...
				return this;
			
			
			// First get the dist matrix
			final LogTimer timer = new LogTimer();
			
			// Do the neighborhood assignments, get sample weights, find core samples..
			final LogTimer neighbTimer = new LogTimer();
			labels = new int[m]; // Initialize labels...
			sampleWeights = new double[m]; // Init sample weights...
			coreSamples = new boolean[m];
			
			
			// Fit the nearest neighbor model...
			final LogTimer rnTimer = new LogTimer();
			final RadiusNeighbors rnModel = new RadiusNeighbors(data,
				new RadiusNeighborsParameters(eps)
					.setSeed(getSeed())
					.setMetric(getSeparabilityMetric())
					.setVerbose(false))
				.fit();
			
			info("fit RadiusNeighbors model in " + rnTimer.toString());
			int[][] nearest = rnModel.getNeighbors().getIndices();
			
			
			int[] ptNeighbs;
			ArrayList<int[]> neighborhoods = new ArrayList<>();
			int numCorePts = 0;
			for(int i = 0; i < m; i++) {
				// Each label inits to -1 as noise
				labels[i] = NOISE_CLASS;
				ptNeighbs = nearest[i];
				
				// Add neighborhood...
				int pts;
				neighborhoods.add(ptNeighbs);
				sampleWeights[i] = pts = ptNeighbs.length;
				coreSamples[i] = pts >= minPts;
				
				if(coreSamples[i]) 
					numCorePts++;
			}
			
			
			// Log checkpoint
			info("completed density neighborhood calculations in " + neighbTimer.toString());
			info(numCorePts + " core point"+(numCorePts!=1?"s":"")+" found");
			
			
			// Label the points...
			int nextLabel = 0, v;
			final Stack<Integer> stack = new Stack<>();
			int[] neighb;
			
			
			LogTimer stackTimer = new LogTimer();
			for(int i = 0; i < m; i++) {
				stackTimer = new LogTimer();
				
				// Want to look at unlabeled OR core points...
				if(labels[i] != NOISE_CLASS || !coreSamples[i])
					continue;
				
		        // Depth-first search starting from i, ending at the non-core points.
		        // This is very similar to the classic algorithm for computing connected
		        // components, the difference being that we label non-core points as
		        // part of a cluster (component), but don't expand their neighborhoods.
				int labelCt = 0;
				while(true) {
					if(labels[i] == NOISE_CLASS) {
						labels[i] = nextLabel;
						labelCt++;
						
						if(coreSamples[i]) {
							neighb = neighborhoods.get(i);
							
							for(i = 0; i < neighb.length; i++) {
								v = neighb[i];
								if(labels[v] == NOISE_CLASS)
									stack.push(v);
							}
						}
					}
					

					if(stack.size() == 0) {
						fitSummary.add(new Object[]{
							nextLabel, labelCt, stackTimer.formatTime(), stackTimer.wallTime()
						});
						
						break;
					}
					
					i = stack.pop();
				}
				
				nextLabel++;
			}
			
			
			// Count missing
			numNoisey = 0;
			for(int lab: labels) if(lab==NOISE_CLASS) numNoisey++;
			
			
			// corner case: numNoisey == m (never gets a fit summary)
			if(numNoisey == m)
				fitSummary.add(new Object[]{
					Double.NaN, 0, stackTimer.formatTime(), stackTimer.wallTime()
				});
			
			
			
			info((numClusters=nextLabel)+" cluster"+(nextLabel!=1?"s":"")+
				" identified, "+numNoisey+" record"+(numNoisey!=1?"s":"")+
					" classified noise");
			
			// Encode to put in order
			labels = new NoiseyLabelEncoder(labels).fit().getEncodedLabels();
			
			sayBye(timer);
			return this;
		}
		
	}// End train
	
	@Override
	public Algo getLoggerTag() {
		return com.clust4j.log.Log.Tag.Algo.DBSCAN;
	}
	
	@Override
	final protected Object[] getModelFitSummaryHeaders() {
		return new Object[]{
			"Cluster #","Num. Core Pts.","Iter. Time","Wall"
		};
	}
	
	@Override
	public int getNumberOfIdentifiedClusters() {
		return numClusters;
	}
	
	@Override
	public int getNumberOfNoisePoints() {
		return numNoisey;
	}
	
	/** {@inheritDoc} */
	@Override
	public int[] predict(RealMatrix newData) {
		final int[] fit_labels = getLabels(); // propagates errors
		final int n = newData.getColumnDimension();
		
		// Make sure matches dimensionally
		if(n != this.data.getColumnDimension())
			throw new DimensionMismatchException(n, data.getColumnDimension());
		
		// Fit a radius model
		RadiusNeighbors radiusModel = 
			new RadiusNeighborsParameters(eps) // no scale necessary; may already have been done
				.setMetric(dist_metric)
				.setSeed(getSeed())
				.fitNewModel(data);
		
		final int[] newLabels = new int[newData.getRowDimension()];
		Neighborhood theHood = radiusModel.getNeighbors(newData);
		
		int[][] indices = theHood.getIndices();
		
		int[] idx_row;
		for(int i = 0; i < indices.length; i++) {
			idx_row = indices[i];
			
			int current_class = NOISE_CLASS;
			if(idx_row.length == 0) { 
				/* 
				 * If there are no indices in this point's radius,
				 * we can just avoid the next step and exit early
				 */
			} else { // otherwise, we know there is something in the radius--noise or other
				int j = 0;
				while(j < idx_row.length) {
					current_class = fit_labels[idx_row[j]];
					
					/*
					 * The indices are ordered ascendingly by dist.
					 * Even if the closest point is a noise point, it
					 * could be within a border point's radius, so we
					 * need to keep going.
					 */
					if(NOISE_CLASS == current_class) {
						j++;
					} else {
						break;
					}
				}
			}
			
			newLabels[i] = current_class;
		}
		
		return newLabels;
	}
}

package com.clust4j.algo;

import java.util.Arrays;
import java.util.Iterator;
import java.util.Map;
import java.util.Random;
import java.util.SortedSet;
import java.util.TreeMap;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.utils.Classifier;
import com.clust4j.utils.ClustUtils;
import com.clust4j.utils.GeometricallySeparable;
import com.clust4j.utils.SupervisedLearner;
import com.clust4j.utils.VecUtils;

public class KNN extends AbstractPartitionalClusterer implements SupervisedLearner, Classifier {

	final private int[] trainLabels;
	final private AbstractRealMatrix test;
	
	private boolean isTrained = false;
	private int[] labels = null;
	
	
	public KNN(AbstractRealMatrix train, AbstractRealMatrix test, final int[] labels, final int k) {
		this(train, test, labels, new KNNPlanner(k));
	}
	
	public KNN(AbstractRealMatrix train, AbstractRealMatrix test, final int[] labels, KNNPlanner planner) {
		super(train, planner, planner.k);
		

		if(labels.length != train.getRowDimension())
			throw new DimensionMismatchException(labels.length, train.getRowDimension());
		if(train.getColumnDimension() != test.getColumnDimension())
			throw new DimensionMismatchException(train.getColumnDimension(), test.getColumnDimension());
		
		this.trainLabels = VecUtils.copy(labels);
		
		if(!planner.scale)
			this.test = (AbstractRealMatrix) test.copy();
		else this.test = super.scale(test, (AbstractRealMatrix) test.copy());
	}
	
	


	public static class KNNPlanner extends AbstractClusterer.BaseClustererPlanner {
		protected GeometricallySeparable dist = DEF_DIST;
		protected Random seed = DEF_SEED;
		protected boolean verbose = DEF_VERBOSE;
		protected boolean scale = DEF_SCALE;
		protected int k;
		
		public KNNPlanner(final int k) {
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
		
		@Override
		public KNNPlanner setSep(final GeometricallySeparable dist) {
			this.dist = dist;
			return this;
		}
		
		@Override
		public KNNPlanner setScale(final boolean scale) {
			this.scale = scale;
			return this;
		}
		
		@Override
		public KNNPlanner setSeed(final Random seed) {
			this.seed = seed;
			return this;
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
		public KNNPlanner setVerbose(final boolean v) {
			this.verbose = v;
			return this;
		}
	}
	
	

	
	@Override
	public int predict(final double[] newRecord) {
		if(newRecord.length != data.getColumnDimension()) {
			if(verbose)
				error("Dimension mismatch: " + newRecord.length + ", " + data.getColumnDimension());
			throw new DimensionMismatchException(newRecord.length, data.getColumnDimension());
		}
		
		TreeMap<Integer, Double> rec_to_dist = new TreeMap<Integer, Double>();
		
		if(verbose)
			info("computing " + k + " nearest neighbors for " + Arrays.toString(newRecord));
		
		// Get map of distances to each record
		for(int train_row = 0; train_row < data.getRowDimension(); train_row++) {
			final double sim = getSeparabilityMetric().getDistance(newRecord, data.getRow(train_row));
			rec_to_dist.put(train_row, sim);
		}
		
		// Sort treemap on value
		// If the distance metric is a similarity metric, we want it DESC else ASC
		SortedSet<Map.Entry<Integer, Double>> sortedEntries = ClustUtils
				.sortEntriesByValue( rec_to_dist );
		return identifyMajorityClass(sortedEntries, k, trainLabels, this);
	}

	@Override
	public int[] getPredictedLabels() {
		return labels;
	}
	
	final static int identifyMajorityClass(SortedSet<Map.Entry<Integer, Double>> sortedEntries, 
			int K, int[] train_labels, KNN knn) {
		TreeMap<Integer, Integer> lab_to_ct = new TreeMap<Integer, Integer>();
		Iterator<Map.Entry<Integer, Double>> iter = sortedEntries.iterator();
		
		if(K == 1) { // Base case... 
			final int label = train_labels[iter.next().getKey()];
			if(knn.verbose)
				knn.info("reached recursion base case; returning label: " + label);
			return label;
		}
		
		int i = 0;
		while(i++ < K) { 
			// We can be certain iter will always have next due to
			// this check in super class: if(k > data.getRowDimension());
			// thus we know k should be less than the number of rows (or entries)
			// in this sortedSet. Additionally, K could be recursively decrementing.
			Map.Entry<Integer, Double> nextEntry = iter.next();
			int correspLabel = train_labels[nextEntry.getKey()];
			
			Integer currCt = lab_to_ct.get(correspLabel);
			if(null == currCt) // Haven't seen label yet
				lab_to_ct.put(correspLabel, 1);
			else 
				lab_to_ct.put(correspLabel, currCt + 1);
		}
		
		
		// Now we have the labels mapped to their counts. Sort
		// descending on value, then deal with ties as necessary
		SortedSet<Map.Entry<Integer, Integer>> desc = ClustUtils.sortEntriesByValue(lab_to_ct, true);
		Iterator<Map.Entry<Integer, Integer>> descIter = desc.iterator();
		Map.Entry<Integer, Integer> first = descIter.next();
		
		Integer maj_ct = first.getValue();  // Holds majority count
		Integer maj_lab = first.getKey();   // Holds majority label
		
		if(descIter.hasNext() && descIter.next().getValue().equals(maj_ct)) { // Then we have a tie...
			if(knn.verbose)
				knn.info("tie identified; recursing with K-1 (" + (K-1) + ") until majority label found");
			return identifyMajorityClass(sortedEntries, K-1, train_labels, knn);
		}
		
		if(knn.verbose)
			knn.info("no ties found; returning label: " + maj_lab);
		return maj_lab;
	}

	@Override
	public boolean isTrained() {
		return isTrained;
	}
	
	public AbstractRealMatrix testSet() {
		return (AbstractRealMatrix) test.copy();
	}

	@Override
	public void train() {
		synchronized(this) { // Must be synch because `isTrained` is a race condition
			if(isTrained)
				return;
			
			final int m = test.getRowDimension();
			final long now = System.currentTimeMillis();
			labels = new int[m];
			
			for(int test_row = 0; test_row < m; test_row++) {
				final double[] test_record = test.getRow(test_row);
				labels[test_row] = predict(test_record);
			}
			
			if(verbose) {
				info("labeling complete. Test labels: " + Arrays.toString(labels));
				info("model " + getKey() + " completed in " + 
						(System.currentTimeMillis() - now)/1000d + " sec");
			}
			
			isTrained = true;
			
		} // End synchronized
	} // End train
	
	@Override
	public int[] truthSet() {
		return VecUtils.copy(trainLabels);
	}
	
	
	
	@Override
	public String getName() {
		return "KNN";
	}
	
	@Override
	public Algo getLoggerTag() {
		return com.clust4j.log.Log.Tag.Algo.KNN;
	}
}

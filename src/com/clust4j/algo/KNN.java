package com.clust4j.algo;

import java.util.Arrays;
import java.util.Iterator;
import java.util.Map;
import java.util.SortedSet;
import java.util.TreeMap;

import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.exception.DimensionMismatchException;

import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.utils.ClustUtils;
import com.clust4j.utils.GeometricallySeparable;
import com.clust4j.utils.Classifier;
import com.clust4j.utils.SupervisedLearner;

public class KNN extends AbstractPartitionalClusterer implements SupervisedLearner, Classifier {
	private boolean isTrained = false;
	private int[] labels = null;
	
	final private AbstractRealMatrix test;
	final private int[] trainLabels;
	
	final public static class KNNPlanner extends AbstractClusterer.BaseClustererPlanner {
		private GeometricallySeparable dist = DEF_DIST;
		private boolean verbose = DEF_VERBOSE;
		private boolean scale = DEF_SCALE;
		private int k;
		
		public KNNPlanner(final int k) {
			this.k = k;
		}
		
		@Override
		public boolean getVerbose() {
			return verbose;
		}
		
		@Override
		public KNNPlanner setDist(final GeometricallySeparable dist) {
			this.dist = dist;
			return this;
		}
		
		@Override
		public KNNPlanner setScale(final boolean scale) {
			this.scale = scale;
			return this;
		}

		@Override
		public GeometricallySeparable getDist() {
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

	public KNN(AbstractRealMatrix train, AbstractRealMatrix test, final int[] labels, final int k) {
		this(train, test, labels, new KNNPlanner(k));
	}
	
	public KNN(AbstractRealMatrix train, AbstractRealMatrix test, final int[] labels, KNNPlanner builder) {
		super(train, builder, builder.k);
		
		if(labels.length != train.getRowDimension())
			throw new DimensionMismatchException(labels.length, train.getRowDimension());
		if(train.getColumnDimension() != test.getColumnDimension())
			throw new DimensionMismatchException(train.getColumnDimension(), test.getColumnDimension());
		
		this.trainLabels = new int[labels.length];
		System.arraycopy(labels, 0, trainLabels, 0, labels.length);
		
		if(!builder.scale)
			this.test = (AbstractRealMatrix) test.copy();
		else this.test = super.scale(test, (AbstractRealMatrix) test.copy());
	}
	
	
	
	
	
	
	
	@Override
	public String getName() {
		return "KNN";
	}

	@Override
	public int[] getPredictedLabels() {
		return labels;
	}
	
	final static int identifyMajorityClass(SortedSet<Map.Entry<Integer, Double>> sortedEntries, int K, int[] train_labels, KNN knn) {
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
		for(int train_row = 0; train_row < data.getRowDimension(); train_row++)
			rec_to_dist.put(train_row, getDistanceMetric().distance(newRecord, data.getRow(train_row)));
		
		// Sort treemap on value
		SortedSet<Map.Entry<Integer, Double>> sortedEntries = ClustUtils.sortEntriesByValue(rec_to_dist);
		return identifyMajorityClass(sortedEntries, k, trainLabels, this);
	}
	
	public AbstractRealMatrix testSet() {
		return (AbstractRealMatrix) test.copy();
	}

	@Override
	public void train() {
		if(isTrained)
			return;
		
		final int m = test.getRowDimension();
		labels = new int[m];
		
		for(int test_row = 0; test_row < m; test_row++) {
			final double[] test_record = test.getRow(test_row);
			labels[test_row] = predict(test_record);
		}
		
		if(verbose)
			info("labeling complete. Test labels: " + Arrays.toString(labels));
		
		isTrained = true;
	}
	
	@Override
	public int[] truthSet() {
		final int[] truthSet = new int[trainLabels.length];
		System.arraycopy(trainLabels, 0, truthSet, 0, trainLabels.length);
		return truthSet;
	}
	
	@Override
	public Algo getLoggerTag() {
		return com.clust4j.log.Log.Tag.Algo.KNN____;
	}
}

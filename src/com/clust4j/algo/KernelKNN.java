package com.clust4j.algo;

import java.util.SortedSet;
import java.util.TreeMap;
import java.util.Map.Entry;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.kernel.AbstractKernel;
import com.clust4j.kernel.LinearKernel;
import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.utils.ClustUtils;
import com.clust4j.utils.GeometricallySeparable;

public class KernelKNN extends AbstractKNNClusterer {
	public static AbstractKernel DEFAULT_KERNEL = new LinearKernel();

	public KernelKNN(AbstractRealMatrix train, AbstractRealMatrix test, final int[] labels, final int k) {
		this(train, test, labels, new KernelKNNPlanner(k, DEFAULT_KERNEL));
	}
	
	public KernelKNN(AbstractRealMatrix train, AbstractRealMatrix test, final int[] labels, KernelKNNPlanner builder) {
		super(train, test, labels, builder);
	}
	
	
	/**
	 * Inner class to enforce Kernel-only distance metrics
	 * 
	 * @author Taylor G Smith
	 */
	public static class KernelKNNPlanner extends KNNPlanner {
		public KernelKNNPlanner(int k, AbstractKernel kern) {
			super(k);
			this.setDist(kern);
		}
		
		@Override
		public KernelKNNPlanner setDist(final GeometricallySeparable dist) {
			if(!(dist instanceof AbstractKernel))
				throw new IllegalArgumentException("distance metric must be instance of AbstractKernel");
			return (KernelKNNPlanner) super.setDist(dist);
		}
		
		@Override
		public KernelKNNPlanner setScale(final boolean scale) {
			return (KernelKNNPlanner) super.setScale(scale);
		}
		
		@Override
		public KernelKNNPlanner setVerbose(final boolean v) {
			return (KernelKNNPlanner) super.setVerbose(v);
		}
	}
	
	/**
	 * Since kernel matrices are a measure of similarity and not
	 * distance, should sort <i>descending</i> instead of asc.
	 */
	@Override
	protected SortedSet<Entry<Integer, Double>> topKSortedByValue(TreeMap<Integer, Double> rec_to_dist) {
		return ClustUtils.sortEntriesByValue(rec_to_dist, true);
	}
	
	
	@Override
	public String getName() {
		return "KernelKNN";
	}
	
	@Override
	public Algo getLoggerTag() {
		return com.clust4j.log.Log.Tag.Algo.KRNLKNN;
	}
}

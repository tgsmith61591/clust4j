package com.clust4j.algo;

import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.log.Log.Tag.Algo;

public class KNN extends AbstractKNNClusterer {

	public KNN(AbstractRealMatrix train, AbstractRealMatrix test, final int[] labels, final int k) {
		this(train, test, labels, new KNNPlanner(k));
	}
	
	public KNN(AbstractRealMatrix train, AbstractRealMatrix test, final int[] labels, KNNPlanner builder) {
		super(train, test, labels, builder);
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

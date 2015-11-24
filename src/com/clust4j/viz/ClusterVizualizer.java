package com.clust4j.viz;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.AbstractRealMatrix;

import com.clust4j.algo.AbstractClusterer;
import com.clust4j.utils.Classifier;

public class ClusterVizualizer<T extends AbstractClusterer & Classifier> {
	private final ClassLabelColorizer colors;
	private final AbstractRealMatrix data;
	private final int[] labels;
	private final int m;
	
	public ClusterVizualizer(T clusterer) {
		this.data = clusterer.getData();
		this.labels = clusterer.getLabels();
		
		if(labels.length != (m=data.getRowDimension()))
			throw new DimensionMismatchException(labels.length, m);
		
		colors = new ClassLabelColorizer(labels);
	}
	
	public void draw() {
		// TODO
	}
	
	public ClassLabelColorizer getClassLabelColors() {
		return colors;
	}
}

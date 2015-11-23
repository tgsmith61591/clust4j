package com.clust4j.algo;

import java.util.Map;
import java.util.SortedSet;
import java.util.TreeMap;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.utils.ClustUtils;
import com.clust4j.utils.GeometricallySeparable;

final class NearestNeighbor {
	private double[] record;
	private int recordIdx;
	private AbstractRealMatrix data;
	private double[][] dist_mat;
	private GeometricallySeparable dist;
	
	
	public NearestNeighbor(final double[] record, 
			final AbstractRealMatrix data, 
			final GeometricallySeparable dist) {
		
		
		if(record.length != data.getColumnDimension())
			throw new DimensionMismatchException(record.length, data.getColumnDimension());
		
		this.record = record;
		this.data = data;
		this.dist = dist;
	}
	
	
	public NearestNeighbor(final int record, final double[][] distMatrix) {
		
		if(distMatrix.length == 0)
			throw new IllegalArgumentException("empty distance matrix");
		if(record >= distMatrix.length || record < 0)
			throw new DimensionMismatchException(record, distMatrix.length);
		
		this.dist_mat = distMatrix;
		this.recordIdx = record;
	}
	
	
	public SortedSet<Map.Entry<Integer, Double>> getSortedNearestWithinRadius(final double rad) {
		// TM container
		TreeMap<Integer, Double> rec_to_dist = new TreeMap<Integer, Double>();
		
		// Get map of distances to each record
		for(int train_row = 0; train_row < dist_mat.length; train_row++) {
			if(train_row == recordIdx)
				continue;
			
			final double sim = dist_mat[FastMath.min(recordIdx, train_row)][FastMath.max(recordIdx, train_row)];
			if(sim < rad) rec_to_dist.put(train_row, sim);
		}
		
		// Sort treemap on value
		// If the distance metric is a similarity metric, we want it DESC else ASC
		return ClustUtils.sortEntriesByValue( rec_to_dist );
	}
	
	
	public SortedSet<Map.Entry<Integer, Double>> getSortedNearest() {
		// TM container
		TreeMap<Integer, Double> rec_to_dist = new TreeMap<Integer, Double>();
		
		// Get map of distances to each record
		for(int train_row = 0; train_row < data.getRowDimension(); train_row++) {
			final double sim = dist.getDistance(record, data.getRow(train_row));
			rec_to_dist.put(train_row, sim);
		}
		
		// Sort treemap on value
		// If the distance metric is a similarity metric, we want it DESC else ASC
		return ClustUtils.sortEntriesByValue( rec_to_dist );
	}
}

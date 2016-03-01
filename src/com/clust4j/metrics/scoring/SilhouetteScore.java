package com.clust4j.metrics.scoring;

import java.util.ArrayList;
import java.util.TreeMap;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.algo.AbstractClusterer;
import com.clust4j.algo.LabelEncoder;
import com.clust4j.metrics.pairwise.GeometricallySeparable;
import com.clust4j.utils.ClustUtils;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;

public class SilhouetteScore implements UnsupervisedEvaluationMetric {
	private final static SilhouetteScore instance = new SilhouetteScore();
	private SilhouetteScore() { }
	
	@Override
	public double evaluate(AbstractRealMatrix data, GeometricallySeparable metric, final int[] labels) {

		double[][] X = data.getData();
		
		final int m = data.getRowDimension();
		if(labels.length != m)
			throw new DimensionMismatchException(m, labels.length);
		
		
		LabelEncoder encoder = null; 
		
		// this method is undefined if numClasses is < 2 or >= m
		try{
			encoder = new LabelEncoder(labels).fit();
		} catch(IllegalArgumentException iae) {
			/*
			model.warn("Silhouette score is undefined "
				+ "for < 2 classes or >= m (" + m + "). "
				+ "Try adjusting parameters within the model "
				+ "to alter the number of clusters found");
			*/
			
			return Double.NaN;
		}
		
		
		
		final int[] encoded    = encoder.getEncodedLabels();
		final int[] uniqueLabs = encoder.getClasses();
		
		
		double[][] distMatrix = ClustUtils.distanceFullMatrix(X, metric);
		double[] intraDists   = VecUtils.rep(1.0, m);
		double[] interDists   = VecUtils.rep(Double.POSITIVE_INFINITY, m);
		
		Integer[] maskIdcs, otherIdxMask;
		double[][] currDists;
		
		
		// To avoid numerous passes on the order of M
		// to get a mask for labels, do it once...
		final int[] uniqueEncoded = encoder.transform(uniqueLabs);
		TreeMap<Integer, Integer[]> labToIdcs = new TreeMap<>();
		for(int label: uniqueEncoded) {
			ArrayList<Integer> ref = new ArrayList<>();
			
			for(int i = 0; i < m; i++) {
				if(encoded[i] == label)
					ref.add(i);
			}
			
			labToIdcs.put(label, ref.toArray(new Integer[ref.size()]));
		}
		
		
		for(int label: uniqueEncoded) {
			
			// Mask of idcs for label
			maskIdcs = labToIdcs.get(label);
			currDists = MatUtils.getRows(distMatrix, maskIdcs);
			

			int numCurrent = maskIdcs.length - 1;
			if(numCurrent != 0) { // i.e., if this isn't the only label for this class
				
				// Get the row sums of all the columns included in the maskIdcs
				for(int idx: maskIdcs) {
					
					// Easy way, but uses too many passes on order of N or M:
					double colSum = 0;
					for(int j = 0; j < currDists.length; j++)
						colSum += currDists[j][idx];
					intraDists[idx] = colSum / numCurrent;
				}
			} // if it does, we need to update inter anyways:
			
			// Look at other labels, see how close other clusters are
			for(int other: uniqueEncoded) {
				if(other == label)
					continue;
				
				else {
					otherIdxMask = labToIdcs.get(other);
					double[] otherDists = MatUtils.rowMeans(
							MatUtils.getColumns(currDists, otherIdxMask));
					
					int k = 0;
					for(int idx: maskIdcs)
						interDists[idx] = FastMath.min(otherDists[k++], 
							interDists[idx]); 
					
				}
			}
		}

		
		// Get difference in distances
		double[] sil = new double[intraDists.length];
		for(int i = 0; i < sil.length; i++)
			sil[i] = (interDists[i] - intraDists[i]) /
				FastMath.max(intraDists[i], interDists[i]);
		
		return VecUtils.mean(sil);
	}

	@Override
	public double evaluate(AbstractClusterer model, GeometricallySeparable metric, final int[] labels) {
		return evaluate(model.getData(), metric, labels);
	}
	
	public static SilhouetteScore getInstance() {
		return instance;
	}
}
